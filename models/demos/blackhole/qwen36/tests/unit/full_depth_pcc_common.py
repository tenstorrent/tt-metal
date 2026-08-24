# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared harness for the full-depth (ALL layers, real weights) logits PCC tests.

``test_prefill.py`` and ``test_decode.py`` both drive the **whole** model — every
layer, the checkpoint's real weights, the shipped bfp8 dtypes — against the
HuggingFace reference ``Qwen3_5ForCausalLM``, and both need the same setup: the
mesh, the full-depth model, a paged KV cache, a deterministic prompt, and the HF
forward to compare against. That setup lives here so the two test files stay short
and cannot drift apart.

Why this exists at all
----------------------
Every other model-level test in this suite compares TT against TT — the contract
path (``prefill_paged`` + the Generator decode chain) against the bespoke
``prefill_tp``/``decode_tp`` oracle — and all of them run a truncated stack
(``n_layers=4`` in the unit tests, ``8`` in the TP ones). That answers "do the two
TT paths agree", never "does the whole stack agree with the reference
implementation": a per-layer bias, a wrong RoPE section split, or a GDN state that
decays slightly wrong is invisible to a TT-vs-TT oracle and is diluted to nothing
when only 4 or 8 of the 32 (or 64) layers run.

One harness, both checkpoints
-----------------------------
Nothing here is model-specific: the checkpoint comes from ``HF_MODEL`` and the mesh
from ``MESH_DEVICE``, exactly like the demo. The same two tests cover

* Qwen3.5-9B  — ``HF_MODEL=Qwen/Qwen3.5-9B``  with ``MESH_DEVICE=P150`` (single
  Blackhole) or ``MESH_DEVICE=N300`` (Wormhole, TP=2), 32 layers
* Qwen3.6-27B — ``HF_MODEL=Qwen/Qwen3.6-27B`` with ``MESH_DEVICE=P150x4`` (or
  ``T3K``), 64 layers

and ``build_full_depth_model`` asserts the stack really is full depth
(``n_layers == num_hidden_layers``), so a stray truncation cannot make either test
pass cheaply.

Reference notes
---------------
* The HF model is the **text-only** ``Qwen3_5ForCausalLM`` built on
  ``Qwen3_5TextConfig`` — the same pair ``Qwen36ModelArgs.load_state_dict`` uses, so
  the composite (3.6 VLM) checkpoint and the text-only (3.5) one both resolve to the
  weights the TT model loaded. ``output_loading_info`` is checked for missing keys:
  a prefix mismatch would otherwise silently leave a randomly-initialised reference
  and the PCC would be measured against noise.
* It runs at the checkpoint's bf16 (``QWEN36_FULL_DEPTH_REF_DTYPE=float32`` to
  upcast), on CPU, with ``use_cache=True`` so a decode step continues from the same
  prompt state the TT decode does.

Measured (Wormhole, bf16 HF reference, 128-token prompt, 5 decode steps)
-----------------------------------------------------------------------
* 9B  / N300 (TP=2), 32 layers: prefill 0.9913; decode 0.9914 0.9929 0.9827
  0.9897 0.9910 (min 0.9827, mean 0.9896)
* 27B / T3K  (TP=8), 64 layers: prefill 0.9957; decode 0.9922 0.9595 0.9876
  0.9909 0.9801 (min 0.9595, mean 0.9821)

The gates in ``pcc_thresholds.json`` sit below the worse of the two with ~1% margin
(prefill 0.98, decode 0.95): one flat function-keyed table serves both checkpoints,
so a per-model number would only be reachable by making the key model-dependent —
which that table deliberately is not. They are REGRESSION DETECTORS at this prompt
and this length, not accuracy targets. A single step's full-vocab PCC over 248320
logits is a coarse instrument (row variance, not device error, moves it), so read
the per-step lines and the argmax/top-5 agreement the tests log, not just the
pass/fail. Decode is the looser mode because each step also carries whatever the GDN
recurrent + conv state and the paged KV accumulated, while prefill is one clean pass.

Env knobs: ``QWEN36_FULL_DEPTH_PROMPT_LEN`` (default 128, keep it a multiple of the
GDN chunk size 128 — the TP prefill runs the chunk-seq kernel over the whole span),
``QWEN36_FULL_DEPTH_DECODE_STEPS`` (default 5), ``QWEN36_FULL_DEPTH_REF_DTYPE``.
"""

import gc
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole, run_for_wormhole_b0_or_blackhole
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE

# Mesh shape from MESH_DEVICE, same table the demo uses (a 9B on Wormhole needs the
# 2-chip N300: a single N150 cannot hold it). Falls back to every visible device,
# capped at 4 — the widest TP the 27B ships with.
MESH_SHAPE = {
    "P150": (1, 1),
    "P150x4": (1, 4),
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (1, 8),
}.get(os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 4)))
_MULTI = MESH_SHAPE != (1, 1)

# Multi-device needs FABRIC_1D for the TP CCLs and the l1_small_size the GDN prefill
# ttnn.conv1d allocates from. Single device never reaches that conv (it runs the MAC
# FIR instead), and on Wormhole that unused 24KB/core reservation is exactly what the
# chunk-seq kernel's static circular buffers collide with — same split
# tests/test_prefill.py makes for the same reason.
_L1_SMALL = GDN_CONV1D_L1_SMALL_SIZE if (_MULTI or is_blackhole()) else 4096
DEVICE_PARAMS = [
    {
        "l1_small_size": _L1_SMALL,
        "num_command_queues": 2,
        **({"fabric_config": ttnn.FabricConfig.FABRIC_1D} if _MULTI else {}),
    }
]

BLOCK_SIZE = 64
# 32 blocks x 64 = 2048 tokens of KV budget, and a multiple of 32 so the chunked-SDPA
# page-table alignment the demo rounds to holds here too.
NUM_BLOCKS = 32

PROMPT_LEN = int(os.environ.get("QWEN36_FULL_DEPTH_PROMPT_LEN", "128"))
DECODE_STEPS = int(os.environ.get("QWEN36_FULL_DEPTH_DECODE_STEPS", "5"))

# Long enough to tokenize past any prompt length these tests use; deterministic text so
# the measured PCCs are comparable run to run.
_PROMPT_TEXT = (
    "The history of computing hardware spans several distinct eras. Mechanical calculators gave way to "
    "relay machines, relays to vacuum tubes, tubes to discrete transistors, and transistors to the "
    "integrated circuit. Each transition changed not only how fast a machine could compute but what kinds "
    "of problems people thought were worth computing at all. The stored-program architecture, in which "
    "instructions and data share one memory, made general-purpose machines practical and turned "
    "programming into a discipline of its own. Modern accelerators return to an older idea: many simple "
    "processing elements operating in parallel on regular data, fed by a memory hierarchy carefully "
    "arranged so that the arithmetic units are rarely idle. The capital of France is "
) * 4


def parametrize_full_depth():
    """The env-selected mesh + the device params both full-depth tests need.

    One decorator rather than three copied ``@parametrize`` lines per file, so the
    two tests cannot end up running on differently-configured devices.
    """

    def decorator(fn):
        fn = pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)(fn)
        fn = pytest.mark.parametrize(
            "mesh_device", [pytest.param(MESH_SHAPE, id=f"{MESH_SHAPE[0]}x{MESH_SHAPE[1]}")], indirect=True
        )(fn)
        return run_for_wormhole_b0_or_blackhole()(fn)

    return decorator


def build_full_depth_model(mesh_device, *, max_seq_len=None, prompt_len=None):
    """Full-depth TT model + tokenizer + a ``[1, prompt_len]`` prompt.

    No ``n_layers`` / ``layer_indices`` truncation, and the depth is asserted rather
    than assumed — a truncated stack would still produce plausible logits and quietly
    turn this into a much weaker test.

    ``max_seq_len`` / ``prompt_len`` default to this module's single-prompt geometry;
    the teacher-forcing e2e test passes its own, since it consumes far more than one
    prompt's worth of positions.
    """
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    mesh_device.enable_program_cache()
    model = Qwen36Model.from_pretrained(
        mesh_device, max_batch_size=1, max_seq_len=max_seq_len or NUM_BLOCKS * BLOCK_SIZE
    )
    args = model.args
    num_hidden_layers = args.hf_config.get_text_config().num_hidden_layers
    assert args.n_layers == num_hidden_layers == len(model.layers), (
        f"not a full-depth stack: n_layers={args.n_layers}, len(model.layers)={len(model.layers)}, "
        f"checkpoint has {num_hidden_layers}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.CKPT_DIR, trust_remote_code=True)
    token_ids = _build_prompt(tokenizer, prompt_len or PROMPT_LEN)
    logger.info(
        f"Full-depth harness: {args.CKPT_DIR} — {args.n_layers} layers, dim={args.dim}, vocab={args.vocab_size}, "
        f"mesh={tuple(mesh_device.shape)} ({model.num_devices} device(s)), prompt={token_ids.shape[1]}"
    )
    return model, tokenizer, token_ids


def _build_prompt(tokenizer, length):
    """Deterministic ``[1, length]`` prompt of exactly ``length`` real tokens (no padding).

    Padding would be wrong here, not merely wasteful: the single-device
    ``prefill_paged`` reads its logit from the LAST row of the sequence it was given
    (``x[:, -1:, :]``), not from ``valid_len - 1``, so a right-padded prompt would
    compare a pad position's logit against HF's real one.
    """
    ids = tokenizer(_PROMPT_TEXT, return_tensors="pt").input_ids
    assert ids.shape[1] >= length, f"prompt text tokenizes to {ids.shape[1]} tokens, need {length}"
    return ids[:, :length].to(torch.long)


def hf_reference(ckpt_dir, token_ids, decode_steps=0):
    """HF prefill + ``decode_steps`` greedy decode steps on CPU, then free the model.

    Returns ``(prefill_logits [vocab], [decode_logits [vocab], ...], [teacher_token, ...])``
    where ``teacher_token[k]`` is the token HF *fed* at decode step ``k`` (its own
    argmax from the previous step), so TT can be driven with the identical inputs and
    step ``k``'s PCC measures step ``k`` rather than the compounding of a greedy
    divergence at step 0. ``decode_steps=0`` returns just the prefill logits.
    """
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    ref_dtype = getattr(torch, os.environ.get("QWEN36_FULL_DEPTH_REF_DTYPE", "bfloat16"))
    logger.info(f"Loading HF reference ({ref_dtype}) from {ckpt_dir} ...")
    text_config = Qwen3_5TextConfig.from_pretrained(ckpt_dir)
    hf_model, loading_info = Qwen3_5ForCausalLM.from_pretrained(
        ckpt_dir, config=text_config, dtype=ref_dtype, output_loading_info=True
    )
    # A composite 3.6 VLM checkpoint carries visual.*/mtp.* the text-only class does not
    # want (unexpected keys are fine); a MISSING key means a weight stayed at its random
    # init and every PCC below would be measured against noise.
    assert not loading_info["missing_keys"], f"HF reference has uninitialized weights: {loading_info['missing_keys']}"
    hf_model.eval()

    with torch.no_grad():
        out = hf_model(token_ids, use_cache=True)
        prefill_logits = out.logits[0, -1].float()
        cache = out.past_key_values

        decode_logits, teacher_tokens = [], []
        tok = int(prefill_logits.argmax())
        for _ in range(decode_steps):
            teacher_tokens.append(tok)
            out = hf_model(torch.tensor([[tok]], dtype=torch.long), past_key_values=cache, use_cache=True)
            step_logits = out.logits[0, -1].float()
            cache = out.past_key_values
            decode_logits.append(step_logits)
            tok = int(step_logits.argmax())

    del hf_model, cache, out
    gc.collect()
    return prefill_logits, decode_logits, teacher_tokens


def allocate_paged_kv(model, num_blocks=NUM_BLOCKS):
    """Paged KV cache + GDN external state; returns the identity page table.

    ``num_blocks`` must cover every position the caller will touch — prompt plus, for
    a decode loop, every step it will take.
    """
    args = model.args
    n_kv = args.n_local_kv_heads if model.num_devices > 1 else args.n_kv_heads
    model.allocate_kv_caches([num_blocks, n_kv, BLOCK_SIZE, args.head_dim], ttnn.bfloat16, batch_size=1)
    return torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)


def tt_prefill_logits(model, token_ids, page_table):
    """Whole prompt through every layer via ``prefill_paged``; returns ``[vocab]`` fp32.

    ``prefill_paged`` is the trusted non-traced path the other prefill tests are
    validated against, and it leaves behind exactly what decode reads: the paged KV of
    the full-attention layers plus each GDN layer's recurrent + conv state.
    """
    vocab = model.args.vocab_size
    tt_logits = model.prefill_paged(token_ids, page_table, valid_len=token_ids.shape[1])
    if model.num_devices > 1:
        # The LM head leaves the logits replicated; read one replica.
        host = ttnn.to_torch(tt_logits, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh_device, dim=0))
    else:
        host = ttnn.to_torch(tt_logits)
    out = host.reshape(-1, host.shape[-1])[0, :vocab].float()
    assert not torch.isnan(out).any(), "NaN in full-depth prefill logits"
    return out


def tt_decode_logits(model, token, position, page_table):
    """One decode step through the vLLM contract path; returns ``[vocab]`` fp32.

    ``prepare_inputs_decode`` → ``ttnn_decode_forward`` → ``process_output_decode`` is
    the chain vLLM and the demo drive, so this gates the shipped decode entry point
    rather than a test-only shortcut.
    """
    dev = model.prepare_inputs_decode(
        torch.tensor([[token]], dtype=torch.int32), torch.tensor([position], dtype=torch.int32), page_table
    )
    tt_out, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
    out = model.process_output_decode(tt_out, 1).reshape(-1)[: model.args.vocab_size].float()
    assert not torch.isnan(out).any(), f"NaN in full-depth decode logits at position {position}"
    return out


def report(label, hf_logits, tt_logits, tokenizer):
    """PCC + argmax/top-5 agreement for one position; returns the PCC.

    The argmax and top-5 lines are the point of logging rather than asserting: a
    full-vocab PCC and a next-token disagreement mean different things, and at 248320
    near-tied logits the second is not by itself a defect.
    """
    _, pcc = comp_pcc(hf_logits, tt_logits, 0.0)
    hf_tok, tt_tok = int(hf_logits.argmax()), int(tt_logits.argmax())
    overlap = len(set(hf_logits.topk(5).indices.tolist()) & set(tt_logits.topk(5).indices.tolist()))
    logger.info(
        f"{label}: PCC={float(pcc):.6f} top5={overlap}/5 argmax HF={hf_tok} ({tokenizer.decode([hf_tok])!r}) "
        f"TT={tt_tok} ({tokenizer.decode([tt_tok])!r}) {'ok' if hf_tok == tt_tok else 'MISMATCH'}"
    )
    return float(pcc)
