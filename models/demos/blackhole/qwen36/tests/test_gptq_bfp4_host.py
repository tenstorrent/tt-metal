# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the GPTQ-BFP4 pipeline library (utils/gptq_bfp4.py). No device, no ttnn.

Run: pytest models/demos/blackhole/qwen36/tests/test_gptq_bfp4_host.py -q
"""
import json
import struct

import pytest
import torch

from models.demos.blackhole.qwen36.utils import gptq_bfp4 as G


def _u32_to_bfp_ref(f, shared_exp, mant_bits):
    """Scalar transcription of blockfloat_common.cpp convert_u32_to_bfp<Bfp*_b, truncate=false> + convert_bfp_to_u32."""
    u = struct.unpack("<I", struct.pack("<f", f))[0]
    mant, exp, sign = u & 0x7FFFFF, (u >> 23) & 0xFF, u >> 31
    if exp == 0:
        return 0.0
    mant |= 1 << 23
    if shared_exp > exp:
        d = shared_exp - exp
        while d > 31:
            mant >>= 31
            d -= 31
        mant >>= d
    S = 24 - mant_bits
    rv = mant & ((1 << S) - 1)
    mant >>= S
    if rv > (1 << (S - 1)) or (rv == (1 << (S - 1)) and (mant & 1)):
        mant += 1
    mant = min(mant, (1 << mant_bits) - 1)
    val = mant * 2.0 ** (shared_exp - 127 - (mant_bits - 1))
    return -val if (sign and mant) else val


def _block_ref(vals, mant_bits):
    exps = [(struct.unpack("<I", struct.pack("<f", v))[0] >> 23) & 0xFF for v in vals]
    e = max(exps)
    return [_u32_to_bfp_ref(v, e, mant_bits) if e else 0.0 for v in vals]


@pytest.mark.parametrize("mant_bits", [3, 7])
def test_bfp_blocks_matches_scalar_reference(mant_bits):
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(64, 16, generator=g) * torch.exp(torch.randn(64, 16, generator=g) * 3)).bfloat16().float()
    x[0] = 0.0  # all-zero block
    x[1, :8] = 0.0
    x[2] = torch.tensor(
        [1.0, 0.875, 0.9375, 0.0625, -0.03125, 1.9375] + [0.0] * 10
    )  # ties + saturation (1.9375 -> m=7)
    x[3, 0] = 1e-40  # fp32 subnormal -> 0, ignored for the max exponent
    got = G.bfp_blocks(x, mant_bits)
    for r in range(x.shape[0]):
        ref = torch.tensor(_block_ref(x[r].tolist(), mant_bits))
        assert torch.equal(got[r], ref), (r, x[r], got[r], ref)


@pytest.mark.parametrize("exp_search", [False, True])
def test_quantize_column_is_ttnn_fixed_point(exp_search):
    g = torch.Generator().manual_seed(1)
    N, K = 16480, 8  # TP8 GDN in-proj stack [qkv 10240; z 6144; a 48; b 48]
    W = torch.randn(N, K, generator=g) * torch.exp(torch.randn(N, 1, generator=g) * 2)
    groups = G.gdn_in_groups(10240, 6144, 48, 8)
    Q = torch.stack([G.quantize_column(W[:, k], groups, exp_search) for k in range(K)], dim=1)
    Qb = Q.bfloat16()
    assert torch.equal(Qb.float(), Q), "grid values must be exact in bf16"
    assert torch.equal(G.ttnn_bfp_roundtrip(Qb, groups), Q), "ttnn re-pack must be the identity"


def test_rtn_equals_ttnn_conversion_of_bf16():
    """Without exp search the quantizer IS ttnn's own rounding on a bf16 weight (RTN = today's BFP4 path)."""
    g = torch.Generator().manual_seed(2)
    W = (torch.randn(256, 64, generator=g) * torch.exp(torch.randn(256, 64, generator=g))).bfloat16()
    groups = G.consecutive_groups(256)
    assert torch.equal(G.rtn_quantize(W.float(), groups), G.ttnn_bfp_roundtrip(W, groups))


def test_gdn_ab_groups_layout():
    g8 = G.gdn_ab_groups(48, 8)
    assert g8.shape == (8, 16)
    assert g8[0].tolist() == [0, 1, 2, 3, 4, 5, 48, 49, 50, 51, 52, 53, -1, -1, -1, -1]
    assert g8[7].tolist()[:12] == list(range(42, 48)) + list(range(90, 96))
    g4 = G.gdn_ab_groups(48, 4)  # TP4: [a12 b4], [b8 pad8] per device (emulation/emul_run.py q_ab)
    assert g4.shape == (8, 16)
    assert g4[0].tolist() == list(range(0, 12)) + [48, 49, 50, 51]
    assert g4[1].tolist() == list(range(52, 60)) + [-1] * 8
    allrows = sorted(r for r in torch.cat([g8.flatten(), g4.flatten()]).tolist() if r >= 0)
    assert allrows == sorted(list(range(96)) * 2)


def test_gdn_ab_grouping_matches_handoff_q_ab_tp4():
    """ttnn_bfp_roundtrip with gdn_ab_groups == the validated TP4 emulation (cat [a_d, b_d, zeros] as [K, 32])."""
    g = torch.Generator().manual_seed(3)
    a, b = torch.randn(48, 40, generator=g).bfloat16(), (torch.randn(48, 40, generator=g) * 7).bfloat16()
    got = G.ttnn_bfp_roundtrip(torch.cat([a, b]), G.gdn_ab_groups(48, 4))
    ra, rb = a.float().clone(), b.float().clone()
    for d in range(4):
        sl = slice(d * 12, (d + 1) * 12)
        M = torch.cat([a[sl].float().t(), b[sl].float().t(), torch.zeros(40, 8)], 1)
        Q = G.bfp_blocks(M.reshape(40, 2, 16), 3).reshape(40, 32)
        ra[sl], rb[sl] = Q[:, :12].t(), Q[:, 12:24].t()
    assert torch.equal(got, torch.cat([ra, rb]))


def test_gdn_in_groups_match_device_layout():
    """Replays gdn/tp.py's fused qkvzab layout (prepare_gdn_qkv head interleave + per-device [qkv|z|a|b] cat +
    tile pad) on real TP8 dims and checks per-device consecutive 16-blocks == gdn_in_groups on the HF stack."""
    tp, nk, dk, nv, dv, K = 8, 16, 128, 48, 128, 4
    key_dim, value_dim = nk * dk, nv * dv
    g = torch.Generator().manual_seed(4)
    qkv = (torch.randn(2 * key_dim + value_dim, K, generator=g) * torch.exp(torch.randn(1, K, generator=g))).bfloat16()
    z, a, b = (
        torch.randn(value_dim, K, generator=g).bfloat16(),
        torch.randn(nv, K, generator=g).bfloat16(),
        torch.randn(nv, K, generator=g).bfloat16(),
    )

    # tp_common.prepare_gdn_qkv, transcribed (it lives in a ttnn-importing module).
    qp, kp, vp = qkv[:key_dim], qkv[key_dim : 2 * key_dim], qkv[2 * key_dim :]
    q_per, v_per = nk // tp, nv // tp
    qkv_re = torch.cat(
        [
            torch.cat(
                [
                    qp[s * q_per * dk : (s + 1) * q_per * dk],
                    kp[s * q_per * dk : (s + 1) * q_per * dk],
                    vp[s * v_per * dv : (s + 1) * v_per * dv],
                ]
            )
            for s in range(tp)
        ]
    )
    qkv_per, z_per, nv_per = (2 * key_dim + value_dim) // tp, value_dim // tp, nv // tp
    dev = []
    for d in range(tp):
        blk = torch.cat(
            [
                qkv_re[d * qkv_per : (d + 1) * qkv_per],
                z[d * z_per : (d + 1) * z_per],
                a[d * nv_per : (d + 1) * nv_per],
                b[d * nv_per : (d + 1) * nv_per],
            ]
        )
        pad = (-blk.shape[0]) % 32
        blk = torch.cat([blk.float(), torch.zeros(pad, K)])
        dev.append(G.ttnn_bfp_roundtrip(blk, G.consecutive_groups(blk.shape[0]))[: blk.shape[0] - pad])
    dev = torch.cat(dev)

    stack = torch.cat([qkv, z, a, b])
    want = G.ttnn_bfp_roundtrip(stack, G.gdn_in_groups(qkv.shape[0], z.shape[0], nv, tp))
    # Map the HF-stack result into the device layout and compare.
    w_qkv, w_z, w_a, w_b = torch.split(want, [qkv.shape[0], value_dim, nv, nv])
    wq, wk, wv = w_qkv[:key_dim], w_qkv[key_dim : 2 * key_dim], w_qkv[2 * key_dim :]
    want_dev = torch.cat(
        [
            torch.cat(
                [
                    wq[d * q_per * dk : (d + 1) * q_per * dk],
                    wk[d * q_per * dk : (d + 1) * q_per * dk],
                    wv[d * v_per * dv : (d + 1) * v_per * dv],
                    w_z[d * z_per : (d + 1) * z_per],
                    w_a[d * nv_per : (d + 1) * nv_per],
                    w_b[d * nv_per : (d + 1) * nv_per],
                ]
            )
            for d in range(tp)
        ]
    )
    assert torch.equal(dev, want_dev)


def test_gptq_beats_rtn_and_is_fixed_point():
    g = torch.Generator().manual_seed(5)
    N, K, T = 64, 96, 2048
    mix = torch.randn(K, K, generator=g) / K**0.5 + torch.eye(K)
    X = torch.randn(T, K, generator=g) @ mix * torch.exp(torch.randn(K, generator=g))  # correlated, uneven scales
    H = X.t() @ X
    W = torch.randn(N, K, generator=g).bfloat16().float()
    groups = G.consecutive_groups(N)
    rtn = G.rtn_quantize(W, groups)
    for act_order in (False, True):
        for exp_search in (False, True):
            Q = G.gptq_quantize(W, H, groups, blocksize=32, act_order=act_order, exp_search=exp_search)
            assert G.proxy_loss(W, Q, H) < 0.8 * G.proxy_loss(W, rtn, H), (act_order, exp_search)
            assert torch.equal(G.ttnn_bfp_roundtrip(Q.bfloat16(), groups), Q)


def test_gptq_dead_columns_keep_rtn_weights():
    g = torch.Generator().manual_seed(6)
    N, K = 32, 48
    X = torch.randn(512, K, generator=g)
    X[:, 7] = 0.0  # input 7 never fires in calibration
    W = torch.randn(N, K, generator=g)
    groups = G.consecutive_groups(N)
    Q = G.gptq_quantize(W, X.t() @ X, groups)
    assert torch.equal(Q[:, 7], G.quantize_column(W[:, 7], groups))  # untouched by error feedback, not zeroed
    assert Q[:, 7].abs().sum() > 0


def test_marker_and_cache_suffix(tmp_path):
    assert G.read_marker(tmp_path) is None and G.cache_suffix(None) == ""
    (tmp_path / G.MARKER_NAME).write_text(json.dumps({"format": G.MARKER_FORMAT, "tp": 8}))
    m = G.read_marker(tmp_path)
    assert m["tp"] == 8 and len(m["_id"]) == 64
    assert G.cache_suffix(m) == "_gptq" + m["_id"][:12]
    (tmp_path / G.MARKER_NAME).write_text(json.dumps({"format": "other"}))
    with pytest.raises(ValueError):
        G.read_marker(tmp_path)


def _tiny_checkpoint(path):
    """Random 4-layer Qwen3.5 text model (3 GDN + 1 attention), TP=2-compatible dims, saved in 3+ shards."""
    q35 = pytest.importorskip("transformers.models.qwen3_5")
    torch.manual_seed(0)
    cfg = q35.Qwen3_5TextConfig(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_num_value_heads=4,
        linear_value_head_dim=32,
        tie_word_embeddings=False,
    )
    model = q35.Qwen3_5ForCausalLM(cfg).to(torch.bfloat16)
    model.save_pretrained(path, max_shard_size="1MB")
    return cfg


def test_pipeline_end_to_end_tiny(tmp_path):
    """quantize -> verify -> kl on a tiny random model, CPU only: the exported checkpoint is a full copy, GPTQ tensors
    are BFP4 fixed points in the TP=2 device layout, everything else is unchanged, the KL stage runs."""
    from models.demos.blackhole.qwen36.scripts import gptq_bfp4 as S

    src, out = tmp_path / "src", tmp_path / "out"
    cfg = _tiny_checkpoint(src)
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (4, 64), generator=g)
    torch.save({"input_ids": ids, "sha256": "test", "sources": {}}, tmp_path / "calib.pt")
    torch.save({"input_ids": ids.flip(0), "sha256": "test2"}, tmp_path / "eval.pt")
    S.main(
        [
            "quantize",
            "--model",
            str(src),
            "--calib",
            str(tmp_path / "calib.pt"),
            "--out",
            str(out),
            "--tp",
            "2",
            "--device",
            "cpu",
            "--blocksize",
            "32",
        ]
    )
    meta = G.read_marker(out)
    assert meta["tp"] == 2 and not (out / ".gptq_parts").exists()
    # 3 GDN layers x (qkv, z, a, b, out) + 4 layers x (gate, up, down)
    assert len(meta["tensors"]) == 3 * 5 + 4 * 3
    assert all(r["rel_err_gptq"] <= r["rel_err_rtn"] for r in meta["stats"])
    with pytest.raises(SystemExit) as e:
        S.main(["verify", "--model", str(src), "--out", str(out)])
    assert e.value.code == 0
    with pytest.raises(SystemExit) as e:
        S.main(
            [
                "kl",
                "--model",
                str(src),
                "--gptq",
                str(out),
                "--eval",
                str(tmp_path / "eval.pt"),
                "--nsamples",
                "2",
                "--skip",
                "4",
                "--chunk",
                "16",
                "--device",
                "cpu",
                "--max-top1-drop",
                "100",
                "--max-kl-vs-rtn",
                "100",
                "--json",
                str(tmp_path / "kl.json"),
            ]
        )
    assert e.value.code == 0
    s = json.loads((tmp_path / "kl.json").read_text())
    assert set(s["variants"]) == {"deployed", "rtn", "gptq"} and s["positions"] == 2 * (64 - 1 - 4)


def test_layer_loop_matches_hf_forward(tmp_path):
    """capture_layer_inputs + run_layer per layer reproduces Qwen3_5ForCausalLM's own forward (masks, M-RoPE kwargs)."""
    from models.demos.blackhole.qwen36.scripts import gptq_bfp4 as S

    cfg = _tiny_checkpoint(tmp_path / "m")
    model, _ = S.load_model(str(tmp_path / "m"))
    ids = torch.randint(0, cfg.vocab_size, (2, 48), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = model(input_ids=ids, use_cache=False).logits
    hs, kws = S.capture_layer_inputs(model, ids, torch.device("cpu"))
    for L, layer in enumerate(model.model.layers):
        hs = S.run_layer(layer, hs, kws[L], torch.device("cpu"), torch.device("cpu"))
    with torch.no_grad():
        got = torch.cat([model.lm_head(model.model.norm(h)) for h in hs])
    assert torch.equal(got, ref)
