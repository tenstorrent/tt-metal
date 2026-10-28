# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device tests: full Muse-Glimmer model (truncated stacks vs torch reference; full-depth generation smoke)."""

import os
import time

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.muse_glimmer_30b.tt.mesh_utils import mesh_test_params, skip_and_submesh
from models.demos.muse_glimmer_30b.tt.model_config import DEFAULT_CKPT_DIR, MuseGlimmerModelArgs

CKPT = os.environ.get("HF_MODEL") or DEFAULT_CKPT_DIR

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(CKPT, "config.json")), reason="checkpoint not available"
)

PCC = 0.98
PROMPT = "What is the capital of France? Answer in one sentence."


_mesh_params = mesh_test_params()


def _text_ids(text: str, n: int, seed: int, vocab: int):
    """Tokenize `text` (tokenizers lib) to n ids; fall back to seeded random ids."""
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(os.path.join(CKPT, "tokenizer.json"))
        ids = tok.encode(text).ids
        while len(ids) < n:
            ids = ids + ids
        return torch.tensor(ids[:n], dtype=torch.long)
    except Exception:
        g = torch.Generator().manual_seed(seed)
        return torch.randint(1000, vocab - 1000, (n,), generator=g)


_TEXT_A = (
    "The history of computing stretches from mechanical calculators to modern accelerators. Early machines "
    "such as the Analytical Engine were designed to follow programs punched on cards, while later electronic "
    "computers used vacuum tubes and then transistors to execute billions of operations every second. "
) * 4
_TEXT_B = (
    "In a quiet valley, a small village sat beside a river that wound its way toward the distant sea. Every "
    "morning the farmers carried baskets of apples and wheat to the market square, where merchants traded "
    "stories along with their goods. "
) * 4


def _chat_prompt_ids():
    """Chat-templated token ids for PROMPT (template rendered via transformers' tokenizer + chat_template.jinja)."""
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    tok_file = os.path.join(CKPT, "tokenizer.json")
    fast = PreTrainedTokenizerFast(tokenizer_file=tok_file, bos_token="<|begin_of_text|>", eos_token="<|end_of_text|>")
    with open(os.path.join(CKPT, "chat_template.jinja")) as f:
        tpl = f.read()
    text = fast.apply_chat_template(
        [{"role": "user", "content": PROMPT}], chat_template=tpl, tokenize=False, add_generation_prompt=True
    )
    # the template already emits <|begin_of_text|>: do not add it again
    return Tokenizer.from_file(tok_file).encode(text, add_special_tokens=False).ids


def _page_table(args, batch):
    nb = args.max_num_blocks_per_user
    return torch.arange(batch * nb, dtype=torch.int32).reshape(batch, nb)


def _page_table_tt(pt, mesh_device):
    return ttnn.from_torch(
        pt,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@_mesh_params
@pytest.mark.parametrize("num_layers", [4, 8])
def test_model_prefill_decode_vs_reference(mesh_device, tp, num_layers):
    sub = skip_and_submesh(mesh_device, tp)
    from models.demos.muse_glimmer_30b.reference.model import CausalLM
    from models.demos.muse_glimmer_30b.tt.model import MuseGlimmerModel

    args = MuseGlimmerModelArgs.from_checkpoint(
        mesh_device=sub, num_layers=num_layers, max_batch_size=2, max_seq_len=1024, max_prefill_len=512
    )
    ref = CausalLM.from_checkpoint(CKPT, num_layers=num_layers, dtype=torch.bfloat16)
    model = MuseGlimmerModel(args, sub)

    prompts = [_text_ids(_TEXT_A, 200, 0, args.vocab_size), _text_ids(_TEXT_B, 150, 1, args.vocab_size)]
    batch = len(prompts)
    pt = _page_table(args, batch)
    pt_tt = _page_table_tt(pt, sub)

    # ---- prefill both users, compare last-token logits
    pasts, ref_next, tt_next = [], [], []
    for u, p in enumerate(prompts):
        with torch.no_grad():
            out = ref(p[None])
        ref_logits = out["logits"][0, -1].float()
        pasts.append(out["past_kvs"])
        tt_logits = model.prefill_forward(p, u, pt_tt)
        assert tt_logits.shape == (args.vocab_size,)
        ok, msg = comp_pcc(ref_logits, tt_logits, PCC)
        top1_ok = int(ref_logits.argmax()) == int(tt_logits.argmax())
        print(f"layers={num_layers} prefill user={u}: {msg} top1_equal={top1_ok} (ref={int(ref_logits.argmax())})")
        assert ok, msg
        ref_next.append(int(ref_logits.argmax()))
        tt_next.append(int(tt_logits.argmax()))

    # ---- 8 teacher-forced decode steps (reference greedy token fed to both)
    positions = [p.shape[0] for p in prompts]
    host_in = model.prepare_decode_inputs_host(torch.tensor(ref_next), torch.tensor(positions), pt)
    dev_in = model.allocate_decode_inputs(batch, pt.shape[1])
    top1_hits = 0
    for step in range(8):
        model.copy_decode_inputs(host_in, dev_in)
        logits_tt = model.decode_forward(*dev_in)
        tt_logits = model.logits_to_torch(logits_tt, 32)
        ttnn.deallocate(logits_tt)
        new_tokens = []
        for u in range(batch):
            with torch.no_grad():
                out = ref(torch.tensor([[ref_next[u]]]), start_pos=positions[u], past_kvs=pasts[u])
            pasts[u] = out["past_kvs"]
            ref_logits = out["logits"][0, -1].float()
            ok, msg = comp_pcc(ref_logits, tt_logits[u], PCC)
            top1_ok = int(ref_logits.argmax()) == int(tt_logits[u].argmax())
            top1_hits += top1_ok
            print(f"layers={num_layers} decode step={step} user={u}: {msg} top1_equal={top1_ok}")
            assert ok, msg
            new_tokens.append(int(ref_logits.argmax()))
        ref_next = new_tokens
        positions = [p + 1 for p in positions]
        host_in = model.prepare_decode_inputs_host(torch.tensor(ref_next), torch.tensor(positions), pt)
    print(f"layers={num_layers} decode top-1 agreement: {top1_hits}/{8 * batch}")


@pytest.mark.slow
@_mesh_params
def test_model_full_depth_smoke(mesh_device, tp):
    sub = skip_and_submesh(mesh_device, tp)
    from tokenizers import Tokenizer

    from models.demos.muse_glimmer_30b.tt.model import MuseGlimmerModel

    args = MuseGlimmerModelArgs.from_checkpoint(mesh_device=sub, max_batch_size=1, max_seq_len=512, max_prefill_len=256)
    assert args.n_layers == 52
    t0 = time.time()
    model = MuseGlimmerModel(args, sub)
    print(f"TIMING model_load={time.time() - t0:.1f}s")

    ids = _chat_prompt_ids()
    n_new = 80
    pt = _page_table(args, 1)
    pt_tt = _page_table_tt(pt, sub)

    t0 = time.time()
    logits = model.prefill_forward(torch.tensor(ids), 0, pt_tt)
    nxt = int(logits.argmax())
    print(f"TIMING prefill ({len(ids)} tok)={time.time() - t0:.3f}s")
    t_dec = time.time()
    generated = [nxt]
    pos = len(ids)
    host_in = model.prepare_decode_inputs_host(torch.tensor([nxt]), torch.tensor([pos]), pt)
    dev_in = model.allocate_decode_inputs(1, pt.shape[1])
    while len(generated) < n_new and nxt not in args.eos_token_ids:
        model.copy_decode_inputs(host_in, dev_in)
        out_tt = model.decode_forward(*dev_in)
        nxt = int(model.logits_to_torch(out_tt, 1)[0].argmax())
        ttnn.deallocate(out_tt)
        generated.append(nxt)
        pos += 1
        host_in = model.prepare_decode_inputs_host(torch.tensor([nxt]), torch.tensor([pos]), pt)

    print(
        f"TIMING decode {len(generated) - 1} steps, {(time.time() - t_dec) / max(1, len(generated) - 1) * 1000:.1f} ms/token (eager, incl. host)"
    )
    text = Tokenizer.from_file(os.path.join(CKPT, "tokenizer.json")).decode(generated, skip_special_tokens=False)
    print(f"generated ids={generated}\ntext={text!r}")
    assert "Paris" in text, text
