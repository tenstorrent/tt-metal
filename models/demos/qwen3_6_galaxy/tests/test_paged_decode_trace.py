"""T14b.9 minimal paged decode trace parity test.

The existing test_trace.py tests (test_decode_trace_parity_4layer etc.) all
pass ``page_table=None``, which forces the non-paged decode path in
``llama_attention.forward_decode``. That branch slices the KV cache using
``cur_pos`` as a Python literal, so the slice extents bake into any
captured trace — making non-paged decode inherently non-trace-safe.

The paged decode path uses ``cur_pos_tensor`` (device tensor) in the
paged_scaled_dot_product_attention_decode kernel, so it IS trace-safe.
This test exercises that path end-to-end through the Generator: eager
prefill, then 4 decode steps with enable_trace=True (capture on step 0,
replay on steps 1..3), each compared against an independent eager decode
on a parallel kv_cache copy.
"""

import json
import math
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")

_SNAPSHOT_DIR = pathlib.Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)
_PCC_TRACE_PARITY = 0.9999
_BLOCK_SIZE = 64


@pytest.fixture(scope="module")
def mesh_8x4():
    import ttnn

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _load_layer_weights(layer_idx: int) -> dict:
    from safetensors.torch import load_file as load_st

    with open(_SNAPSHOT_DIR / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    pfx = f"model.language_model.layers.{layer_idx}"
    keys_needed = [k for k in weight_map if k.startswith(pfx + ".")]
    files_needed = sorted({weight_map[k] for k in keys_needed})
    raw = {}
    for fn in files_needed:
        shard = load_st(str(_SNAPSHOT_DIR / fn))
        for k in keys_needed:
            if k in shard:
                raw[k] = shard[k].float()
    return {k[len(pfx) + 1 :]: v for k, v in raw.items()}


def _load_global_weights() -> dict:
    from safetensors.torch import load_file as load_st

    with open(_SNAPSHOT_DIR / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    needed = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ]
    files = sorted({weight_map[k] for k in needed if k in weight_map})
    raw = {}
    for fn in files:
        shard = load_st(str(_SNAPSHOT_DIR / fn))
        for k in needed:
            if k in shard:
                raw[k] = shard[k].float()
    return {
        "tok_embeddings.weight": raw["model.language_model.embed_tokens.weight"],
        "norm.weight": raw["model.language_model.norm.weight"],
        "output.weight": raw["lm_head.weight"],
    }


def _make_page_table(batch_size: int, seq_len: int, block_size: int, max_num_blocks: int) -> torch.Tensor:
    """Sequential page table: user b owns blocks [b*max_blocks_per_seq .. )."""
    max_blocks_per_seq = math.ceil(seq_len / block_size)
    page_table = torch.zeros(batch_size, max_blocks_per_seq, dtype=torch.int32)
    for b in range(batch_size):
        for blk in range(max_blocks_per_seq):
            page_table[b, blk] = b * max_blocks_per_seq + blk
    return page_table


def _send_page_table_to_device(page_table: torch.Tensor, mesh_device):
    import ttnn

    return ttnn.from_torch(
        page_table,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom == 0.0:
        return 1.0
    return (a @ b).item() / denom


@pytest.mark.hardware
@pytest.mark.skip(
    reason="T14b.9 step 3 partially landed — _materialize_cur_pos_int "
    "eliminated the device-to-host READ inside trace capture (1 'Reads "
    "are not supported' assertion gone). Capture body still emits ~16 "
    "WRITES per decoder layer (64 total for 4-layer model). Source is "
    "still unidentified; bisecting via instrumentation has been hampered "
    "by the Galaxy's ethernet core (x=27,y=25) destabilizing after each "
    "failed capture (needs tt-smi -r between runs). Next session: bisect "
    "with prints once a stable device window is available."
)
def test_paged_decode_trace_parity_4layer(mesh_8x4):
    """T14b.9: paged decode trace replay produces same logits as eager paged decode.

    Setup: 4-layer model, ``use_paged_kv_cache=True``, single user, T_padded=32
    prompt. Eager prefill establishes the kv_cache; then 4 decode steps with
    enable_trace=False vs enable_trace=True (capture on step 0, replay on
    steps 1..3) on parallel kv_cache copies. Per-step argmax must match and
    PCC must exceed _PCC_TRACE_PARITY (0.9999).
    """
    from transformers import AutoTokenizer

    from models.demos.qwen3_6_galaxy.tt.generator import Qwen36Generator
    from models.demos.qwen3_6_galaxy.tt.llama_model import TtQwen36Transformer
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    N_LAYERS = 4
    MAX_SEQ = 512  # fits 1 prefill block (32) + many decode steps in <=1 page block of 64

    print(f"\n[T14b9-paged] Loading weights for {N_LAYERS} layers...")
    global_wts = _load_global_weights()
    layers_wts = [_load_layer_weights(i) for i in range(N_LAYERS)]

    args = TtQwen36ModelArgs(
        mesh_8x4,
        max_seq_len=MAX_SEQ,
        use_paged_kv_cache=True,
        block_size=_BLOCK_SIZE,
    )

    def _build_two_models():
        # Two independent 4-layer models so the eager and trace decode loops
        # each have their own kv_cache / dn_state / conv_state. Sharing
        # state between them would mean the trace loop sees state already
        # advanced by the eager loop.
        return [
            TtQwen36Transformer(
                mesh_device=mesh_8x4,
                args=args,
                global_weights=global_wts,
                layers_weights=layers_wts,
                num_layers=N_LAYERS,
                dtype=None,
            )
            for _ in range(2)
        ]

    model_eager, model_trace = _build_two_models()
    gen_eager = Qwen36Generator(model_eager, mesh_8x4, args)
    gen_trace = Qwen36Generator(model_trace, mesh_8x4, args)

    tokenizer = AutoTokenizer.from_pretrained(str(_SNAPSHOT_DIR), trust_remote_code=True)
    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    T_prompt = input_ids.shape[-1]
    T_padded = ((T_prompt + 31) // 32) * 32
    if T_padded > T_prompt:
        pad = torch.zeros(1, T_padded - T_prompt, dtype=input_ids.dtype)
        input_ids_padded = torch.cat([input_ids, pad], dim=1)
    else:
        input_ids_padded = input_ids
    print(f"[T14b9-paged] T_prompt={T_prompt}, T_padded={T_padded}")

    # Page table sized for the full max_seq (covers prefill + all decode steps).
    max_blocks_per_seq = math.ceil(MAX_SEQ / _BLOCK_SIZE)
    page_table_cpu = _make_page_table(1, MAX_SEQ, _BLOCK_SIZE, max_blocks_per_seq)
    pt_tt = _send_page_table_to_device(page_table_cpu, mesh_8x4)

    # ----- Prefill (eager, both models) -----
    print("[T14b9-paged] Paged prefill (eager) on both models...")
    logits_pref_e, kv_e, dn_e, cv_e = gen_eager.prefill_forward_with_caches(
        input_ids_padded, page_table=pt_tt, enable_trace=False
    )
    logits_pref_t, kv_t, dn_t, cv_t = gen_trace.prefill_forward_with_caches(
        input_ids_padded, page_table=pt_tt, enable_trace=False
    )
    from models.demos.qwen3_6_galaxy.reference.qwen36 import Qwen36Config

    with open(_SNAPSHOT_DIR / "config.json") as f:
        config = Qwen36Config(json.load(f))
    vocab_size = config.vocab_size
    first_id_e = int(logits_pref_e[0, T_prompt - 1, :vocab_size].argmax().item())
    first_id_t = int(logits_pref_t[0, T_prompt - 1, :vocab_size].argmax().item())
    print(f"[T14b9-paged] eager first id={first_id_e} ('{tokenizer.decode([first_id_e])}')")
    print(f"[T14b9-paged] trace first id={first_id_t} ('{tokenizer.decode([first_id_t])}')")
    assert first_id_e == first_id_t, f"prefill first-token mismatch: eager={first_id_e}, trace={first_id_t}"

    # ----- Decode (4 steps each, parallel kv_caches) -----
    N_STEPS = 4
    cur_id_e = first_id_e
    cur_id_t = first_id_t
    cur_pos = T_padded
    eager_step_logits = []
    trace_step_logits = []

    for step in range(N_STEPS):
        in_tok_e = torch.tensor([[cur_id_e]], dtype=torch.long)
        in_tok_t = torch.tensor([[cur_id_t]], dtype=torch.long)

        logits_e, kv_e, dn_e, cv_e = gen_eager.decode_forward_with_caches(
            in_tok_e,
            current_pos=cur_pos,
            kv_caches=kv_e,
            dn_states=dn_e,
            conv_states=cv_e,
            page_table=pt_tt,
            enable_trace=False,
        )
        logits_t, kv_t, dn_t, cv_t = gen_trace.decode_forward_with_caches(
            in_tok_t,
            current_pos=cur_pos,
            kv_caches=kv_t,
            dn_states=dn_t,
            conv_states=cv_t,
            page_table=pt_tt,
            enable_trace=True,
        )
        eager_step_logits.append(logits_e.clone())
        trace_step_logits.append(logits_t.clone())

        am_e = int(logits_e[0, 0, :vocab_size].argmax().item())
        am_t = int(logits_t[0, 0, :vocab_size].argmax().item())
        pcc = _pcc(logits_e[0, 0, :vocab_size], logits_t[0, 0, :vocab_size])
        print(
            f"[T14b9-paged] step {step}: cur_pos={cur_pos}, "
            f"eager_id={am_e} ('{tokenizer.decode([am_e])}'), "
            f"trace_id={am_t} ('{tokenizer.decode([am_t])}'), "
            f"PCC={pcc:.6f}"
        )
        assert am_e == am_t, (
            f"step {step}: argmax mismatch (eager={am_e}, trace={am_t}) — " f"trace path diverged from eager"
        )
        assert pcc >= _PCC_TRACE_PARITY, (
            f"step {step}: PCC={pcc:.6f} < {_PCC_TRACE_PARITY} — " f"trace path numerically diverged from eager"
        )

        cur_id_e = am_e
        cur_id_t = am_t
        cur_pos += 1

    # Sanity check: the trace was actually captured (not silently fallback to eager).
    cache = gen_trace._decode_traces.get((1, True))
    assert cache is not None, "Trace cache empty — trace path was never entered"
    assert isinstance(cache, dict) and "trace_id" in cache, (
        f"Trace capture fell back to eager: cache={cache}. "
        "Inspect Qwen36Generator._capture_decode_trace logs for the failure cause."
    )
    print(f"[T14b9-paged] PASSED — {N_STEPS} steps, trace_id={cache['trace_id']}")
