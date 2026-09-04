# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Spec decode must be deterministic: same prompt, same model, same tokens every time.

Greedy decoding has no sampling, so two runs of the same request MUST emit identical ids and the
same acceptance rate. This caught a real race: the speculative verify writes the K+1 candidates'
KV with one batched paged_update_cache, but all candidates belong to ONE sequence, so their
consecutive positions share a physical block and several cores read-modify-write the SAME 32-row
tile. Runs then differed in acceptance (2.82 vs 2.61) and diverged from plain greedy at a different
token each time.

Parametrized over a BLOCK_SIZE-aligned prompt (128) and an UNALIGNED one (130). The unaligned length
is the regression guard for the seed forward's KV placement: the eager seed ran through the
masked-bucket prefill's block-aligned paged_fill_cache, so at an anchor that is not a multiple of
BLOCK_SIZE the seed token's K/V landed (anchor % BLOCK_SIZE) slots early and clobbered real prompt KV
in that block. A 128-token prompt divides evenly and hides the bug entirely.

Run: MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/tests/test_spec_determinism.py -v -s
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS, _get_prompt
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

RUNS = 3
MAX_NEW = 48


def _prompt_of_len(target, tokenizer):
    """Exactly `target` token ids, deterministically. _get_prompt already repeats-and-clips the
    shared 128-token prompt for any length <= 256, but truncate defensively so an unaligned target
    (130) is guaranteed exact — that exactness is the whole point of the 130 case."""
    ids = _get_prompt(target, tokenizer)
    while ids.shape[1] < target:  # belt-and-braces: repeat, then clip
        ids = torch.cat([ids, ids], dim=1)
    ids = ids[:, :target]
    assert ids.shape[1] == target, f"prompt is {ids.shape[1]} tokens, wanted exactly {target}"
    return ids


# 128 is BLOCK_SIZE-aligned; 130 is not (130 % 64 == 2) and exercises the unaligned-anchor seed KV write.
@run_for_blackhole()
@pytest.mark.parametrize("prompt_len", [128, 130])
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_decode_is_deterministic(mesh_device, prompt_len):
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.tt.spec_decode import SpeculativeDecoder

    device = mesh_device
    device.enable_program_cache()
    num_blocks = 64
    model = Qwen36Model.from_pretrained(device, max_batch_size=1, max_seq_len=num_blocks * BLOCK_SIZE)
    assert model.mtp is not None, "MTP head not built"
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    token_ids = _prompt_of_len(prompt_len, tokenizer)
    prompt_ids = token_ids[0].tolist()
    kv_shape = [num_blocks, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    pt = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)

    outs, accepts = [], []
    for r in range(RUNS):
        model.free_kv_caches()
        model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
        dec = SpeculativeDecoder(model, pt)
        outs.append(dec.generate(prompt_ids, MAX_NEW))
        accepts.append(dec.accept_rate())
        logger.info(f"[determinism] prompt_len={prompt_len} run {r}: accept={accepts[-1]:.3f}/{dec.K}")
        model.free_kv_caches()

    for r in range(1, RUNS):
        div = next((i for i in range(MAX_NEW) if outs[r][i] != outs[0][i]), None)
        assert div is None, (
            f"run {r} diverged from run 0 at token {div} (prompt_len={prompt_len}) — "
            f"spec decode is nondeterministic.\n"
            f"run0={outs[0]}\nrun{r}={outs[r]}"
        )
    assert (
        len(set(f"{a:.6f}" for a in accepts)) == 1
    ), f"acceptance varied across runs (prompt_len={prompt_len}): {accepts}"
    logger.info(f"[determinism] prompt_len={prompt_len}: {RUNS} runs identical, accept={accepts[0]:.3f}")


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_sampling_is_deterministic(mesh_device):
    """Spec sampling must reproduce from its seed and actually depend on it. The accept path (tt/spec_sampling.py) draws every uniform from one seeded CPU torch.Generator (K per iteration), so same prompt + params + seed must yield the same tokens AND the same accept rate.
    Equal tokens with a different accept rate would mean rejection landed at a different depth and recovery merely agreed (leaked global RNG, or sampler/device state surviving between decoders); reproducing at a different seed would mean the sampler ignored its seed or degenerated to greedy — so run C requires the trajectory to move.
    prompt_len is the unaligned 130 only; aligned 128 is already covered for greedy by test_spec_decode_is_deterministic.
    """
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.tt.spec_decode import SpeculativeDecoder
    from models.demos.blackhole.qwen36.tt.spec_sampling import SpecSamplingParams

    prompt_len = 130
    device = mesh_device
    device.enable_program_cache()
    num_blocks = 64
    model = Qwen36Model.from_pretrained(device, max_batch_size=1, max_seq_len=num_blocks * BLOCK_SIZE)
    assert model.mtp is not None, "MTP head not built"
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    token_ids = _prompt_of_len(prompt_len, tokenizer)
    prompt_ids = token_ids[0].tolist()
    kv_shape = [num_blocks, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    pt = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)

    def run(tag, seed):
        # Same reset recipe as the greedy test above: free + reallocate the paged KV caches and
        # build a fresh decoder; the GDN recurrent state is re-zeroed inside the prefill.
        model.free_kv_caches()
        model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
        dec = SpeculativeDecoder(
            model, pt, sampling=SpecSamplingParams(temperature=1.0, top_k=20, top_p=0.95, seed=seed)
        )
        out = dec.generate(prompt_ids, MAX_NEW)
        rate = dec.accept_rate()
        logger.info(f"[sampling-det] run {tag} seed={seed}: accept={rate:.3f}/{dec.K} tokens={out}")
        dec.log_stats(prefix="sampling-det")
        model.free_kv_caches()
        assert len(out) == MAX_NEW, f"run {tag} (seed={seed}) produced {len(out)} tokens, wanted {MAX_NEW}"
        return out, rate

    a, a_rate = run("A", 1234)
    b, b_rate = run("B", 1234)
    c, c_rate = run("C", 1235)

    div_ab = next((i for i in range(MAX_NEW) if b[i] != a[i]), None)
    assert div_ab is None, (
        f"runs A and B share seed 1234 but diverged at token {div_ab} — spec sampling is not "
        f"reproducible from its seed.\nA={a}\nB={b}"
    )
    assert f"{a_rate:.6f}" == f"{b_rate:.6f}", (
        f"runs A and B share seed 1234 and emitted identical tokens, but acceptance differs "
        f"({a_rate} vs {b_rate}) — the rejection landed at a different depth."
    )
    logger.info(f"[sampling-det] seed 1234 reproducible: A == B for all {MAX_NEW} tokens, accept={a_rate:.3f}")

    div_ac = next((i for i in range(MAX_NEW) if c[i] != a[i]), None)
    assert div_ac is not None, (
        f"run C used seed 1235 yet emitted exactly the seed-1234 tokens for all {MAX_NEW} tokens — "
        f"the sampler is ignoring its seed (or this path degenerated to greedy).\nA={a}\nC={c}"
    )
    logger.info(
        f"[sampling-det] seed 1235 first diverges from seed 1234 at token {div_ac}: "
        f"{a[div_ac]} ({tokenizer.decode([a[div_ac]])!r}) vs {c[div_ac]} "
        f"({tokenizer.decode([c[div_ac]])!r}); accept A={a_rate:.3f} B={b_rate:.3f} C={c_rate:.3f}"
    )
