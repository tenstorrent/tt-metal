# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3.1 FP64 chain golden producer (``golden_producer.py``) on tiny data: each FP64 piece against an
independent reference (vendored AttnRes read, model reference norm / FFN / MoE / MLA attention, DeepSeek-V3.2 indexer),
the segmented chain against a one-shot reference walk, its golden read back through the reader and resumed bit for bit,
and the checkpoint name mapping. CPU only; no ttnn."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from models.demos.common.prefill.long_context_accuracy.fp64_golden import open_golden
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.model import IndexerCPU, ModelArgs
from models.demos.deepseek_v3_d_p.reference.kimi_k3.modeling_kimi_k3_mla import KimiMLAAttention, KimiRMSNorm
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3_1 import golden_producer as producer
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3_1.golden import KimiK31ChainGolden


def _write_trace(path: Path, streams: dict[str, torch.Tensor], chunk_rows: int) -> None:
    """``streams`` as a chunked_group_a_v1 trace in ``path``, in ``chunk_rows``-row shards."""
    index = {}
    for key, value in streams.items():
        rows = value.shape[0]
        chunks = []
        for lo in range(0, rows, chunk_rows):
            hi = min(lo + chunk_rows, rows)
            rel = f"{key}/rows_{lo:08d}_{hi:08d}.safetensors"
            (path / key).mkdir(parents=True, exist_ok=True)
            save_file({key: value[lo:hi].contiguous()}, str(path / rel))
            chunks.append({"row_start": lo, "row_end": hi, "path": rel})
        index[key] = {
            "row_count": rows,
            "dtype": str(value.dtype),
            "shape_tail": list(value.shape[1:]),
            "chunks": chunks,
        }
    (path / "index.json").write_text(json.dumps({"tensor_streams": index}))
    (path / "metadata.json").write_text(json.dumps({"n_prompt_tokens": next(iter(streams.values())).shape[0]}))


SMALL = producer.Dims(heads=4, nope=8, rope=4, v=6, lora=16, index_heads=4, index_dim=16, topk=6, eps=1e-5)


def _weights(dims: producer.Dims, hidden: int, q_lora: int, seed: int = 5) -> dict[str, torch.Tensor]:
    """Random FP64 weights of the layer's attention (checkpoint names), scaled by fan-in."""
    gen = torch.Generator().manual_seed(seed)
    shapes = {
        "q_a_proj.weight": (q_lora, hidden),
        "q_a_layernorm.weight": (q_lora,),
        "q_b_proj.weight": (dims.heads * (dims.nope + dims.rope), q_lora),
        "kv_a_proj_with_mqa.weight": (dims.lora + dims.rope, hidden),
        "kv_a_layernorm.weight": (dims.lora,),
        "kv_b_proj.weight": (dims.heads * (dims.nope + dims.v), dims.lora),
        "g_proj.weight": (dims.heads * dims.v, hidden),
        "o_proj.weight": (hidden, dims.heads * dims.v),
        "indexer.wq_b.weight": (dims.index_heads * dims.index_dim, q_lora),
        "indexer.wk.weight": (dims.index_dim, hidden),
        "indexer.k_norm.weight": (dims.index_dim,),
        "indexer.k_norm.bias": (dims.index_dim,),
        "indexer.weights_proj.weight": (dims.index_heads, hidden),
    }
    assert set(shapes) == set(producer.DSA_WEIGHTS)
    out = {}
    for name, shape in shapes.items():
        t = torch.randn(*shape, generator=gen, dtype=torch.float64)
        out[name] = 1 + 0.1 * t if len(shape) == 1 and not name.endswith("bias") else t / shape[-1] ** 0.5
    return out


def _small_layer(seq: int = 20, hidden: int = 32, q_lora: int = 24):
    w = _weights(SMALL, hidden, q_lora)
    x = torch.randn(seq, hidden, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    return w, x, producer.layer_keys(x, w, SMALL), _queries(x, w, SMALL)


def _queries(x: torch.Tensor, w: dict, dims) -> dict[str, torch.Tensor]:
    """``producer.layer_queries`` plus the index query and head weights, as the chain computes them per block."""
    q = producer.layer_queries(x, w, dims)
    qr = producer.rms64(x.double() @ w["q_a_proj.weight"].T, w["q_a_layernorm.weight"], dims.eps)
    q["index_q"] = (qr @ w["indexer.wq_b.weight"].T).reshape(-1, dims.index_heads, dims.index_dim)
    q["index_w"] = (x.double() @ w["indexer.weights_proj.weight"].T) * dims.index_heads**-0.5 * dims.index_dim**-0.5
    return q


def _index_scores(index_q: torch.Tensor, index_w: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """One row's indexer score of each key, row-wise: ``sum_h relu(q_h . k) w_h``; ``index_q`` [H, D], ``index_w``
    [H]."""
    return torch.relu(index_q @ keys.T).T @ index_w


def test_fp64_indexer_matches_the_deepseek_v32_reference():
    """The golden's indexer score and top-k equal `cpu_deepseek_v32.IndexerCPU` (NoPE, no Hadamard/fp8: the K3.1
    configuration) on random weights. The reference's LayerNorm and score einsum run in fp32, so scores agree to fp32
    rounding (a semantic difference is O(1)), and the top-k sets match exactly on every row with a full selection."""
    seq, hidden, q_lora = 20, 32, 24
    w, x, keys, q = _small_layer(seq, hidden, q_lora)
    args = ModelArgs(
        max_batch_size=1,
        max_seq_len=seq,
        dim=hidden,
        q_lora_rank=q_lora,
        index_n_heads=SMALL.index_heads,
        index_head_dim=SMALL.index_dim,
        index_topk=SMALL.topk,
        index_rope_dim=0,
        index_k_norm_eps=SMALL.eps,
    )
    ref = IndexerCPU(args, use_fp8_path=False)
    ref.wq_b.weight = torch.nn.Parameter(w["indexer.wq_b.weight"])
    ref.wk.weight = torch.nn.Parameter(w["indexer.wk.weight"])
    ref.k_norm.weight = torch.nn.Parameter(w["indexer.k_norm.weight"].float())
    ref.k_norm.bias = torch.nn.Parameter(w["indexer.k_norm.bias"].float())
    ref.weights_proj.weight = torch.nn.Parameter(w["indexer.weights_proj.weight"].float())
    ref.k_cache = torch.zeros(1, seq, SMALL.index_dim, dtype=torch.float64)  # the bf16 buffer would round the keys
    qr = producer.rms64(x @ w["q_a_proj.weight"].T, w["q_a_layernorm.weight"], SMALL.eps)
    mask = torch.full((seq, seq), float("-inf")).triu(1)
    with torch.no_grad():
        ref_top, ref_scores = ref(x.unsqueeze(0), qr.unsqueeze(0), 0, None, mask)
    for r in range(seq):
        got = _index_scores(q["index_q"][r], q["index_w"][r], keys["index_key"][: r + 1])
        want = ref_scores[0, r, : r + 1].double()
        assert (got - want).abs().max() <= 1e-5 * want.abs().max(), r
        if r + 1 >= SMALL.topk:
            assert set(got.topk(SMALL.topk).indices.tolist()) == set(ref_top[0, r].tolist()), r


def test_fp64_attention_matches_the_kimi_reference_layer(monkeypatch):
    """The golden's absorbed attention over each row's selection equals the unabsorbed model reference
    (`KimiMLAAttention`: per-head k/v from kv_b_proj, NoPE k_rot expanded, SDPA, gate before o_proj) masked to the
    same selection, in FP64, to 1e-12 NRMSE. The reference's RMSNorm upcasts to fp32; it runs in FP64 here, with the
    model's eps (1e-5, as vLLM and the device; the trimmed reference defaults to 1e-6)."""
    seq, hidden, q_lora = 20, 32, 24
    w, x, keys, q = _small_layer(seq, hidden, q_lora)

    def rms_fp64(self, h):
        return self.weight * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))

    monkeypatch.setattr(KimiRMSNorm, "forward", rms_fp64)
    config = SimpleNamespace(
        hidden_size=hidden,
        num_attention_heads=SMALL.heads,
        num_key_value_heads=SMALL.heads,
        q_lora_rank=q_lora,
        qk_rope_head_dim=SMALL.rope,
        kv_lora_rank=SMALL.lora,
        v_head_dim=SMALL.v,
        qk_nope_head_dim=SMALL.nope,
        mla_use_nope=True,
        mla_use_output_gate=True,
    )
    ref = KimiMLAAttention(config).double().eval()
    ref.load_state_dict({n: w[n] for n in producer.DSA_WEIGHTS if not n.startswith("indexer.")})
    ref.q_a_layernorm.variance_epsilon = ref.kv_a_layernorm.variance_epsilon = SMALL.eps
    mask = torch.full((1, 1, seq, seq), float("-inf"), dtype=torch.float64)
    selections = []
    for r in range(seq):
        scores = _index_scores(q["index_q"][r], q["index_w"][r], keys["index_key"][: r + 1])
        selections.append(scores.topk(min(SMALL.topk, r + 1)).indices)
        mask[0, 0, r, selections[-1]] = 0.0
    with torch.no_grad():
        want = ref(x.unsqueeze(0), attention_mask=mask)[0]
    for r, sel in enumerate(selections):
        got = producer.attend(q, r, keys["latent"][sel], keys["kpe"][sel], w, SMALL)
        assert ((got - want[r]).norm() / want[r].norm()) <= 1e-12, r


class _F64Torch:
    """``torch`` as the model reference modules see it, with ``float32`` read as ``float64``: the references hard-code
    fp32 compute, and the FP64 comparison substitutes the dtype without editing their semantics."""

    float32 = torch.float64

    def __getattr__(self, name):
        return getattr(torch, name)


@pytest.fixture
def fp64_references(monkeypatch):
    """Run the vendored AttnRes read, `KimiRMSNorm` and the trimmed MoE reference in FP64."""
    from models.demos.deepseek_v3_d_p.reference.kimi_k3 import modeling_kimi_moe

    monkeypatch.setattr(modeling_kimi_moe, "torch", _F64Torch())
    monkeypatch.setattr(
        KimiRMSNorm,
        "forward",
        lambda self, h: self.weight * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)),
    )
    return monkeypatch


@contextmanager
def _fp64_upcasts(monkeypatch):
    """`Tensor.float` as `double` while the context lasts: the vendored `_apply_attn_res` upcasts with it. Only
    around reference code, so the code under test keeps its own casts."""
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "float", lambda self, *a, **k: self.double())
        yield


def _rel(got: torch.Tensor, want: torch.Tensor) -> float:
    return float((got.double() - want.double()).norm() / want.double().norm())


def _tiny_moe_cfg(hidden: int):
    from models.demos.deepseek_v3_d_p.reference.kimi_k3.configuration_kimi_k3 import KimiLinearConfig

    return KimiLinearConfig(
        hidden_size=hidden,
        routed_expert_hidden_size=16,
        moe_intermediate_size=8,
        intermediate_size=24,
        num_experts=6,
        num_experts_per_token=2,
        num_shared_experts=2,
        num_expert_group=1,
        topk_group=1,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        routed_scaling_factor=KimiK3Config.ROUTE_SCALE,
        latent_moe_use_norm=True,
        hidden_act="situ",
        activation_situ_beta=KimiK3Config.ACTIVATION_SITU_BETA,
        activation_situ_linear_beta=KimiK3Config.ACTIVATION_SITU_LINEAR_BETA,
        rms_norm_eps=KimiK3Config.RMS_NORM_EPS,
    )


def _moe64_of(block, top_k: int):
    from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3_1.golden_producer import MoE64

    w = {n: p.detach() for n, p in block.named_parameters() if not n.startswith("experts.")}
    experts = [tuple(getattr(x, m).weight.detach() for m in ("w1", "w3", "w2")) for x in block.experts]
    return MoE64(w, experts, top_k, block.routed_expert_norm.variance_epsilon)


def test_fp64_attn_res_matches_the_vendored_read(fp64_references):
    """`attn_res64` (scores as proj . rms_norm(candidate), a softmax mix of the raw candidates) equals the vendored
    HuggingFace read run in FP64 to <= 1e-14 relative, with one and two sealed snapshots; mixing the normalized
    candidates instead (a plausible slip) is O(1) away."""
    from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.hf_walk import hf_attn_res
    from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3_1.golden_producer import attn_res64, rms64

    gen = torch.Generator().manual_seed(1)
    d = 32
    norm = 1 + 0.1 * torch.randn(d, generator=gen, dtype=torch.float64)
    proj = torch.randn(1, d, generator=gen, dtype=torch.float64)
    for s in (1, 2):
        live = torch.randn(5, d, generator=gen, dtype=torch.float64)
        sealed = 3 * torch.randn(5, s, d, generator=gen, dtype=torch.float64)
        with _fp64_upcasts(fp64_references):
            want = hf_attn_res(live, sealed, norm, proj, 1e-5)
        assert want.dtype == torch.float64
        assert _rel(attn_res64(live, sealed, (norm, proj), 1e-5), want) <= 1e-14
        v = torch.cat([sealed, live.unsqueeze(1)], 1)
        probs = (rms64(v, norm, 1e-5) @ proj[0]).softmax(-1)
        assert _rel(torch.einsum("ns,nsd->nd", probs, rms64(v, norm, 1e-5)), want) > 0.1


def test_fp64_dense_ffn_and_moe_match_the_model_reference(fp64_references):
    """`dense_mlp64` equals the reference `KimiMLP` and `MoE64` (experts grouped, each expert's rows at once) the
    reference `KimiSparseMoeBlock` (per expert over sorted tokens, then a weighted sum per row), all in FP64: outputs
    to <= 1e-13 relative, the routing's sets and weights to rounding, the selection best first with a positive margin
    to the next expert. Mutations (shared experts left out; the routing weights not renormalized) are O(0.01-1)."""
    from models.demos.deepseek_v3_d_p.reference.kimi_k3.modeling_kimi_moe import KimiMLP, KimiSparseMoeBlock
    from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3_1.golden_producer import dense_mlp64

    torch.manual_seed(5)
    hidden = 32
    cfg = _tiny_moe_cfg(hidden)
    block = KimiSparseMoeBlock(cfg).double().eval()
    with torch.no_grad():
        block.gate.e_score_correction_bias.normal_(std=0.05)
        block.routed_expert_norm.weight.normal_(1.0, 0.1)
    x = torch.randn(11, hidden, dtype=torch.float64)
    with torch.no_grad():
        ref_ids, ref_w = block.gate(x.unsqueeze(0))
        want = block(x.unsqueeze(0))[0]
    moe = _moe64_of(block, 2)
    got = moe(x)
    assert _rel(got["output"], want) <= 1e-13
    order = ref_ids.argsort(dim=1)
    assert torch.equal(got["ids"].sort(dim=1).values, ref_ids.sort(dim=1).values)
    assert torch.allclose(got["weights"].gather(1, got["ids"].argsort(dim=1)), ref_w.gather(1, order), rtol=1e-14)
    choice = (x @ block.gate.weight.T).sigmoid() + block.gate.e_score_correction_bias
    assert (choice.gather(1, got["ids"]).diff(dim=1) <= 0).all() and (got["margin"] > 0).all()
    shared = {
        f"{n}.weight": getattr(block.shared_experts, n).weight.detach() for n in ("gate_proj", "up_proj", "down_proj")
    }
    assert _rel(got["output"] - dense_mlp64(x, shared), want) > 0.01
    with torch.no_grad():
        assert _rel(block.shared_experts(x), dense_mlp64(x, shared)) <= 1e-14
        dense = KimiMLP(cfg).double().eval()
        w = {f"{n}.weight": getattr(dense, n).weight.detach() for n in ("gate_proj", "up_proj", "down_proj")}
        assert _rel(dense_mlp64(x, w), dense(x)) <= 1e-14
    unnormalized = _moe64_of(block, 2)
    unnormalized.route = lambda h, r=moe.route: (lambda i, w, m: (i, w * 3, m))(*r(h))
    assert _rel(unnormalized(x)["output"], want) > 0.01


def test_index_topk_matches_the_rowwise_golden():
    """The batched indexer (`index_topk`, key blocks of 7 here) scores every causal key as the row-wise
    score (`_index_scores`) does to <= 1e-14 relative and selects the same top-k, best first; a row with fewer
    causal keys than top-k selects all of them and pads with -1 (score NaN); keys after a row are never selected."""

    gen = torch.Generator().manual_seed(2)
    keys = torch.randn(40, 16, generator=gen, dtype=torch.float64)
    q = torch.randn(6, 4, 16, generator=gen, dtype=torch.float64)
    w = torch.randn(6, 4, generator=gen, dtype=torch.float64)
    rows = torch.tensor([0, 3, 9, 10, 25, 38])
    producer.KEY_BLOCK, block = 7, producer.KEY_BLOCK
    try:
        idx, score = producer.index_topk(q, w, keys, rows, 10)
    finally:
        producer.KEY_BLOCK = block
    for i, r in enumerate(rows.tolist()):
        want = _index_scores(q[i], w[i], keys[: r + 1])
        top = want.topk(min(10, r + 1))
        n = top.indices.numel()
        assert torch.equal(idx[i, :n], top.indices) and (idx[i, n:] == -1).all()
        assert ((score[i, :n] - top.values).abs() <= 1e-14 * want.abs().max()).all() and score[i, n:].isnan().all()
        assert int(idx[i].max()) <= r


def _tiny_chain(hidden: int = 32, vocab: int = 50):
    """A ChainModel with random weights at tiny dimensions, and the reference modules it was built from."""
    from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
    from models.demos.deepseek_v3_d_p.reference.kda.tests.helpers import random_weights
    from models.demos.deepseek_v3_d_p.reference.kimi_k3.modeling_kimi_moe import KimiMLP, KimiSparseMoeBlock
    from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3_1.golden_producer import ChainModel

    torch.manual_seed(9)
    gen = torch.Generator().manual_seed(9)
    eps = KimiK3Config.RMS_NORM_EPS
    kda_cfg = KDAConfig(hidden, 2, 8, 8, 4, eps, use_full_rank_gate=True, gate_lower_bound=-5.0)
    vec = lambda: 1 + 0.1 * torch.randn(hidden, generator=gen, dtype=torch.float64)  # noqa: E731
    query = lambda: (vec(), torch.randn(1, hidden, generator=gen, dtype=torch.float64))  # noqa: E731
    cfg = _tiny_moe_cfg(hidden)
    refs = {"mlp": KimiMLP(cfg).double().eval()}
    layers = {}
    for l in range(4):
        layers[l] = {
            "input_layernorm": vec(),
            "attn_res": query(),
            "mlp_res": query(),
            "post_attention_layernorm": vec(),
        }
        if l < 3:
            layers[l]["kda"] = {n: v.double() for n, v in random_weights(kda_cfg).items()}
    layers[0]["mlp"] = {
        f"{n}.weight": getattr(refs["mlp"], n).weight.detach() for n in ("gate_proj", "up_proj", "down_proj")
    }
    for l in (1, 2):
        block = KimiSparseMoeBlock(cfg).double().eval()
        with torch.no_grad():
            block.gate.e_score_correction_bias.normal_(std=0.05)
            block.routed_expert_norm.weight.normal_(1.0, 0.1)
        refs[l] = block
        layers[l]["moe"] = _moe64_of(block, 2)
    layers[3]["dsa"] = _weights(SMALL, hidden, 24)
    layers[3]["gate"] = torch.randn(6, hidden, generator=gen, dtype=torch.float64)
    layers[3]["gate_bias"] = 0.05 * torch.randn(6, generator=gen, dtype=torch.float64)
    embedding = torch.randn(vocab, hidden, generator=gen).to(torch.bfloat16)
    model = ChainModel(embedding, layers, kda_cfg, SMALL, eps, top_k=2)
    return model, refs


def _reference_walk(model, refs, token_ids: torch.Tensor, monkeypatch) -> dict:
    """Layers 0-3 over the whole sequence at once, by the vendored AttnRes walk (`hf_walk.hf_layer`), the reference
    norm and MoE modules, one-shot FP64 KDA per layer, and the DSA golden's row-wise indexer and attention."""
    from models.demos.deepseek_v3_d_p.reference.kda.layer import kda_forward_reference
    from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.hf_walk import HfStream, hf_layer

    def norm(weight):
        n = KimiRMSNorm(weight.numel(), eps=model.eps).double()
        n.weight = torch.nn.Parameter(weight)
        return n

    got = {}
    stream = HfStream(model.embedding[token_ids].double(), model.block_size, model.eps)
    for l in range(3):
        L = model.layers[l]

        def attn_fn(h, L=L, l=l):
            out = kda_forward_reference(
                norm(L["input_layernorm"])(h)[None], L["kda"], model.kda_config, dtype=torch.float64
            )[0][0]
            got[f"attn_output_{l}"] = out
            return out

        def mlp_fn(h, L=L, l=l):
            x = norm(L["post_attention_layernorm"])(h)
            out = refs[l](x.unsqueeze(0))[0] if l else refs["mlp"](x)
            got[f"moe_input_{l}"], got[f"moe_output_{l}"] = x, out
            return out

        with torch.no_grad(), _fp64_upcasts(monkeypatch):
            hf_layer(stream, l, L["attn_res"], L["mlp_res"], attn_fn, mlp_fn)
        if l:
            got[f"routing_ids_{l}"] = refs[l].gate(got[f"moe_input_{l}"].unsqueeze(0))[0]
    L = model.layers[3]
    with _fp64_upcasts(monkeypatch):
        got["dsa_input_3"] = stream.read(L["attn_res"])
    x = norm(L["input_layernorm"])(got["dsa_input_3"]).detach()
    keys = producer.layer_keys(x, L["dsa"], SMALL)
    q = _queries(x, L["dsa"], SMALL)
    got.update(keys)
    got["topk"] = [
        _index_scores(q["index_q"][r], q["index_w"][r], keys["index_key"][: r + 1]).topk(min(SMALL.topk, r + 1))
        for r in range(x.shape[0])
    ]
    got["attn_output_3"] = torch.stack(
        [
            producer.attend(q, r, keys["latent"][t.indices], keys["kpe"][t.indices], L["dsa"], SMALL)
            for r, t in enumerate(got["topk"])
        ]
    )
    return got


IDENTITY = {"model": "tiny", "token_source": "random ids"}


def _files(run_dir: Path, index: int) -> dict:
    """Segment ``index``'s two files, flattened to ``{"segment.<key>" | "state.<key>.<...>": value}``."""

    def flat(d, pre):
        out = {}
        for k, v in d.items():
            out.update(flat(v, f"{pre}{k}.") if isinstance(v, dict) else {f"{pre}{k}": v})
        return out

    return {
        **flat(torch.load(producer.segment_path(run_dir, index), weights_only=True), "segment."),
        **flat(torch.load(producer.state_path(run_dir, index), weights_only=True), "state."),
    }


def _same(a: dict, b: dict) -> bool:
    """Bit-equal flattened files (NaN where the other has NaN)."""
    return a.keys() == b.keys() and all(
        (torch.equal(v, b[k]) or (v.isnan() == b[k].isnan()).all() and torch.equal(v.nan_to_num(), b[k].nan_to_num()))
        if torch.is_tensor(v)
        else v == b[k]
        for k, v in a.items()
    )


def test_fp64_chain_segments_match_a_one_shot_reference_walk_and_resume(tmp_path: Path, fp64_references):
    """The segmented chain (10 rows in segments 4 / 4 / 2: KDA state and DSA keys carried across, a last segment
    shorter than the others) equals a one-shot walk of independent forms in FP64 (vendored AttnRes read and seal
    schedule, reference norm / FFN / MoE modules, whole-sequence KDA, row-wise indexer and attention) to <= 1e-12
    relative on every product, with identical routing and top-k, read back through the registry and the chain
    reader; the cross-checks against a trace built from that walk read exact embeddings, PCC ~1 and recall 1. A run
    stopped after 2 segments and resumed writes the same files bit for bit."""

    model, refs = _tiny_chain()
    rows = 10
    token_ids = torch.randint(0, 50, (rows,), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        want = _reference_walk(model, refs, token_ids, fp64_references)
    trace_dir = tmp_path / "trace"
    topk_idx = torch.full((rows, SMALL.topk), -1, dtype=torch.int32)
    for r, t in enumerate(want["topk"]):
        topk_idx[r, : t.indices.numel()] = t.indices.int()
    streams = {
        "decoder_input_layer_0": model.embedding[token_ids],
        "attn_output_layer_0": want["attn_output_0"].to(torch.bfloat16),
        "decoder_output_layer_0": want["moe_output_0"].to(torch.bfloat16),
        "expert_ids_layer_1": want["routing_ids_1"].int(),
        "expert_ids_layer_2": want["routing_ids_2"].int(),
        "expert_ids_layer_3": torch.zeros(rows, 2, dtype=torch.int32),
        "kv_post_transform_layer_3": torch.cat([want["latent"], want["kpe"]], -1).to(torch.bfloat16),
        "indexer_k_layer_3": want["index_key"].to(torch.bfloat16),
        "dsa_topk_indices_layer_3": topk_idx,
    }
    _write_trace(trace_dir, streams, chunk_rows=4)
    sampled = producer.sample_rows(rows, 4)

    def run(golden_id: str, max_segments: int | None = None) -> dict:
        with torch.no_grad():
            return producer.run_chain(
                run_dir=tmp_path / golden_id,
                model=model,
                token_ids=token_ids,
                segment=4,
                sampled=sampled,
                identity=IDENTITY,
                trace=producer.Trace(trace_dir),
                max_segments=max_segments,
            )

    manifest = run("chain")
    assert [(s["start"], s["end"]) for s in manifest["segments"]] == [(0, 4), (4, 8), (8, 10)]
    assert manifest["identity"] == {**IDENTITY, "weights_sha256": {}} and manifest["rows"] == rows
    golden = open_golden("chain", tmp_path, reader=KimiK31ChainGolden)
    assert torch.equal(golden.token_ids(), token_ids)
    kinds = {n: s["kind"] for n, s in manifest["products"].items()}
    assert {n for n, k in kinds.items() if k == "sampled_row"} == {"sampled_rows", "attn_output_3"}
    assert {n for n, k in kinds.items() if k == "segment_end"} == {
        f"kda.{l}.{s}" for l in range(3) for s in ("recurrent", "q_convolution", "k_convolution", "v_convolution")
    }
    names = ["moe_input_1", "moe_output_1", "moe_input_2", "moe_output_2", "dsa_input_3", "latent", "kpe", "index_key"]
    got = golden.read_rows(names + ["routing_ids_1", "routing_ids_2", "topk_indices_3"], 0, rows)
    for n in names:
        assert got[n].dtype == (torch.float64 if n in producer.KEYS else torch.float32), n
        assert manifest["products"][n]["dtype"] == str(got[n].dtype).removeprefix("torch."), n
    for n in producer.KEYS:
        assert _rel(got[n], want[n]) <= 1e-12, n
    for n in names[:5]:  # stored fp32: equal to the walk at fp32 rounding (the FP64 values are checked below)
        assert _rel(got[n], want[n]) <= 1e-7, n
    for l in (1, 2):
        assert torch.equal(got[f"routing_ids_{l}"].long().sort(1).values, want[f"routing_ids_{l}"].sort(1).values)
    assert torch.equal(got["topk_indices_3"], topk_idx)
    sampled_products = golden.read_sampled()
    assert sampled_products["sampled_rows"].tolist() == sampled
    attn = sampled_products["attn_output_3"]
    assert attn.dtype == torch.float64 and _rel(attn, want["attn_output_3"][sampled]) <= 1e-12
    # FP64 products before the fp32 store: rerun one segment's chain in memory from the stored carried state
    carried = producer.load_carried(tmp_path / "chain", 1)
    products, _ = producer.chain_segment(model, token_ids[4:8], 4, carried, sampled)
    for n in names[:5]:
        assert _rel(products[n], want[n][4:8]) <= 1e-12, n
    checks = manifest["segments"][1]["checks"]
    assert checks["embedding_ulp_mismatches"] == 0 and checks["routing_recall_1"] == 1.0
    assert (
        checks["kv_latent_pcc"] > 0.9999 and checks["attn_output_0_pcc"] > 0.9999 and checks["topk_recall_mean"] == 1.0
    )
    assert manifest["bytes_per_row"]["topk_indices_3"] == 4 * SMALL.topk

    partial = run("bounded", max_segments=2)
    assert len(partial["segments"]) == 2 and not producer.segment_path(tmp_path / "bounded", 2).exists()
    resumed = run("bounded")
    assert [s["first_segment"] for s in resumed["sessions"]] == [0, 2] and len(resumed["segments"]) == 3
    for index in range(3):
        assert _same(_files(tmp_path / "chain", index), _files(tmp_path / "bounded", index)), index


def test_chain_model_reads_each_checkpoint_tensor_into_its_slot(tmp_path: Path):
    """`ChainModel.from_checkpoint` on a tiny checkpoint whose every tensor is distinct puts each one where its
    checkpoint name says: per layer the pre-attention (`self_attention_res_*`) and pre-FFN (`mlp_res_*`) queries, the
    two norms, layer 0-2 KDA, layer 0's dense FFN, layer 1-2 MoE (non-routed weights and every routed expert as
    (w1, w3, w2)), layer 3's DSA attention and gate; the KDA config, eps and expert count come from config.json."""
    from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
    from models.demos.deepseek_v3_d_p.reference.kda.tests.helpers import random_weights
    from models.demos.deepseek_v3_d_p.reference.kda.weights import normalize_kda_state_dict

    hidden, experts, eps = 16, 3, 1e-5
    kda_cfg = KDAConfig(hidden, 2, 8, 8, 4, eps, use_full_rank_gate=True, gate_lower_bound=-5.0)
    gen = torch.Generator().manual_seed(3)
    rnd = lambda *shape: torch.randn(*shape, generator=gen).to(torch.bfloat16)  # noqa: E731
    t = {"embed_tokens.weight": rnd(20, hidden)}
    for l in range(4):
        p = f"layers.{l}."
        for n in ("input_layernorm", "post_attention_layernorm", "mlp_res_norm", "self_attention_res_norm"):
            t[f"{p}{n}.weight"] = rnd(hidden)
        t[f"{p}mlp_res_proj.weight"], t[f"{p}self_attention_res_proj.weight"] = rnd(1, hidden), rnd(1, hidden)
        if l < 3:
            t.update(
                {
                    f"{p}self_attn.{n}": (v + torch.randn(v.shape, generator=gen) * 0.01).to(torch.bfloat16)
                    for n, v in random_weights(kda_cfg).items()
                }
            )
        if l == 0:
            t.update({f"{p}mlp.{n}.weight": rnd(8, hidden) for n in ("gate_proj", "up_proj", "down_proj")})
        if l >= 1:
            t[f"{p}block_sparse_moe.gate.weight"], t[f"{p}block_sparse_moe.gate.e_score_correction_bias"] = rnd(
                experts, hidden
            ), rnd(experts)
        if l in (1, 2):
            for n in producer.MoE_NON_ROUTED[2:]:
                t[f"{p}block_sparse_moe.{n}"] = rnd(4, 4)
            for e in range(experts):
                t.update({f"{p}block_sparse_moe.experts.{e}.{m}.weight": rnd(4, 4) for m in ("w1", "w3", "w2")})
        if l == 3:
            t.update({f"{p}self_attn.{n}": rnd(5, 3) for n in producer.DSA_WEIGHTS})
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    save_file({f"model.{k}": v.contiguous() for k, v in t.items()}, str(ckpt / "shard.safetensors"))
    (ckpt / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {f"model.{k}": "shard.safetensors" for k in t}})
    )
    linear = {
        "num_heads": 2,
        "head_dim": 8,
        "short_conv_kernel_size": 4,
        "use_full_rank_gate": True,
        "gate_lower_bound": -5.0,
    }
    config = {"hidden_size": hidden, "rms_norm_eps": eps, "num_experts": experts, "linear_attn_config": linear}
    (ckpt / "config.json").write_text(json.dumps({"text_config": config}))
    model = producer.ChainModel.from_checkpoint(ckpt, threads=2, log=lambda m: None)

    same = lambda got, name: torch.equal(got, t[name].double())  # noqa: E731
    assert torch.equal(model.embedding, t["embed_tokens.weight"]) and model.eps == eps and model.kda_config == kda_cfg
    for l in range(4):
        L, p = model.layers[l], f"layers.{l}."
        assert same(L["input_layernorm"], f"{p}input_layernorm.weight")
        assert same(L["post_attention_layernorm"], f"{p}post_attention_layernorm.weight")
        assert same(L["attn_res"][0], f"{p}self_attention_res_norm.weight") and same(
            L["attn_res"][1], f"{p}self_attention_res_proj.weight"
        )
        assert same(L["mlp_res"][0], f"{p}mlp_res_norm.weight") and same(L["mlp_res"][1], f"{p}mlp_res_proj.weight")
        if l < 3:
            kda = {n[len(p + "self_attn.") :]: v for n, v in t.items() if n.startswith(p + "self_attn.")}
            want = normalize_kda_state_dict(kda, kda_cfg)
            assert want.keys() == L["kda"].keys() and all(torch.equal(L["kda"][n], want[n].double()) for n in want)
        if l == 0:
            assert all(
                same(L["mlp"][f"{n}.weight"], f"{p}mlp.{n}.weight") for n in ("gate_proj", "up_proj", "down_proj")
            )
        if l in (1, 2):
            moe = L["moe"]
            assert all(same(moe.w[n], f"{p}block_sparse_moe.{n}") for n in producer.MoE_NON_ROUTED)
            assert len(moe.experts) == experts
            for e in range(experts):
                for got, m in zip(moe.experts[e], ("w1", "w3", "w2")):
                    assert torch.equal(got, t[f"{p}block_sparse_moe.experts.{e}.{m}.weight"]), (l, e, m)
        if l == 3:
            assert all(same(L["dsa"][n], f"{p}self_attn.{n}") for n in producer.DSA_WEIGHTS)
            assert same(L["gate"], f"{p}block_sparse_moe.gate.weight")
            assert same(L["gate_bias"], f"{p}block_sparse_moe.gate.e_score_correction_bias")
    assert set(model.digests) == {
        "embed_tokens",
        "layers.0",
        "layers.1",
        "layers.2",
        "layers.3",
        "layers.1.experts",
        "layers.2.experts",
    }
