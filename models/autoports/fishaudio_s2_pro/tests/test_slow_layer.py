"""Functional-decoder test (tt-model-bringup stage 1 analogue): ONE slow-tower layer on TTNN vs fish-speech's
own torch TransformerBlock on identical inputs, plus the host frame-embedding path and the norm+LM-head tail.
Real weights. Prints PCC per stage so a convention bug is localised, not just detected.

Env: FISH_S2_MESH_SHAPE (1x1), FISH_S2_LAYERS (default 1), FISH_S2_SEQ (default 32,127).
"""
import math
import os

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

from models.autoports.fishaudio_s2_pro.config import S2Config  # noqa: E402
from models.autoports.fishaudio_s2_pro.reference import llama_ref as FR  # noqa: E402
from models.autoports.fishaudio_s2_pro.tt import weights as W  # noqa: E402
from models.autoports.fishaudio_s2_pro.tt.device import open_mesh, parse_shape  # noqa: E402
from models.autoports.fishaudio_s2_pro.tt.prompt import S2Tokenizer, build_prompt  # noqa: E402
from models.autoports.fishaudio_s2_pro.tt.slow_model import S2SlowTransformer  # noqa: E402

N_LAYERS = int(os.environ.get("FISH_S2_LAYERS", 1))


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.fixture(scope="module")
def handle():
    h = open_mesh(parse_shape(os.environ.get("FISH_S2_MESH_SHAPE", "1x1")))
    yield h
    h.close()


@pytest.fixture(scope="module")
def setup(handle, snapshot):
    from models.tt_transformers.tt.model_config import ModelArgs

    cfg = S2Config.from_snapshot(snapshot)
    sd = W.load_fish_state_dict(snapshot)
    slow = W.slow_tower_state_dict(sd, cfg)
    cb = W.codebook_table(sd, cfg)
    view = W.ensure_hf_view(snapshot)
    os.environ["HF_MODEL"] = str(view)
    os.environ.setdefault("TT_CACHE_PATH", os.path.expanduser("~/.cache/fish_s2_pro/tt_cache"))
    args = ModelArgs(handle.mesh, instruct=False, max_batch_size=1, max_seq_len=4096)
    args.n_layers = N_LAYERS
    model = S2SlowTransformer(
        args, ttnn.bfloat8_b, handle.mesh, slow, args.weight_cache_path(ttnn.bfloat8_b), codebook_table=cb, s2cfg=cfg
    )
    # fish torch reference for the same layers (fp32)
    fcfg = FR.BaseModelArgs.from_pretrained(str(snapshot))
    fcfg.max_seq_len = 4096
    layers = []
    for i in range(N_LAYERS):
        blk = FR.TransformerBlock(fcfg, use_sdpa=True)
        blk.load_state_dict(
            {
                k[len(f"text_model.model.layers.{i}.") :]: v.float()
                for k, v in sd.items()
                if k.startswith(f"text_model.model.layers.{i}.")
            },
            strict=True,
        )
        blk.attention.kv_cache = FR.KVCache(1, 4096, fcfg.n_local_heads, fcfg.head_dim, dtype=torch.float32)
        layers.append(blk.eval())
    freqs_cis = FR.precompute_freqs_cis(4096, fcfg.head_dim, fcfg.rope_base).float()
    causal = torch.tril(torch.ones(4096, 4096, dtype=torch.bool))
    return dict(
        cfg=cfg,
        sd=sd,
        slow=slow,
        cb=cb,
        args=args,
        model=model,
        layers=layers,
        freqs_cis=freqs_cis,
        causal=causal,
        fcfg=fcfg,
    )


def _rotate_half_variant(x, freqs_cis):
    """HF (rotate-half) RoPE on the same cos/sin — used only to identify a convention mismatch."""
    cos = freqs_cis[..., 0].repeat_interleave(2, dim=-1)  # (S, D)
    sin = freqs_cis[..., 1].repeat_interleave(2, dim=-1)
    # rotate-half pairs (i, i+D/2)
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    rot = torch.cat([-x2, x1], dim=-1)
    S = x.shape[1]
    return x * cos[:S].view(1, S, 1, -1) + rot * sin[:S].view(1, S, 1, -1)


@pytest.mark.parametrize("seq", [int(s) for s in os.environ.get("FISH_S2_SEQ", "32,127").split(",")])
def test_prefill_layers_vs_fish(setup, snapshot, seq):
    cfg, model, args = setup["cfg"], setup["model"], setup["args"]
    tok = S2Tokenizer(snapshot)
    text = "The quick brown fox jumps over the lazy dog. " * 8
    prompt = build_prompt(tok, text)[:, :seq]
    # make some positions semantic so the codebook path is exercised
    g = torch.Generator().manual_seed(0)
    n_sem = max(1, seq // 4)
    prompt[0, -n_sem:] = torch.randint(cfg.semantic_begin_id, cfg.semantic_end_id + 1, (n_sem,), generator=g)
    prompt[1, -n_sem:] = prompt[0, -n_sem:] - cfg.semantic_begin_id
    prompt[2:, -n_sem:] = torch.randint(0, 1024, (cfg.num_codebooks - 1, n_sem), generator=g)
    T = prompt.shape[1]
    from models.tt_transformers.tt.common import get_padded_prefill_len

    S = get_padded_prefill_len(T)
    tokens = torch.nn.functional.pad(prompt[0], (0, S - T)).view(1, S)
    model.set_frame_codes(prompt[1:], offset=0)

    # --- host embedding vs fish embed
    x_host = model.embed_frames_host(tokens.view(-1), model._codes_for(0, S)).float()  # (S, D)
    ref = FR.DualARTransformer.__new__(FR.DualARTransformer)  # only for math below; use tables directly
    E = setup["slow"]["tok_embeddings.weight"].float()
    CB = setup["cb"].float()
    codes = torch.nn.functional.pad(prompt[1:], (0, S - T))
    fish_x = E[tokens.view(-1)]
    vq = sum(CB[codes[i] + i * cfg.codebook_size] for i in range(cfg.num_codebooks))
    mask = (tokens.view(-1) >= cfg.semantic_begin_id) & (tokens.view(-1) <= cfg.semantic_end_id)
    fish_x = torch.where(mask.view(-1, 1), (fish_x + vq) / math.sqrt(cfg.num_codebooks + 1), fish_x)
    p_embed = pcc(x_host[:T], fish_x[:T])
    print(f"\n[seq {seq}] host embedding vs fish embed: PCC {p_embed:.6f}")

    # --- TT: prepare inputs (our override) and run the layer stack, get raw residual out (get_last_token=-1)
    inputs = model.prepare_inputs_prefill(tokens, page_table=None, last_token_idx=T - 1)
    x_tt_in = ttnn.to_torch(inputs[0]).float().reshape(-1, args.dim)
    print(
        f"[seq {seq}] TT input residual vs host embedding: PCC {pcc(x_tt_in[:T], x_host[:T]):.6f}  shape {tuple(inputs[0].shape)}"
    )
    out = model.ttnn_prefill_forward(
        inputs[0], rot_mats_global=inputs[1], rot_mats_local=inputs[2], user_id=0, page_table=None, get_last_token=-1
    )
    h_tt = ttnn.to_torch(out).float().reshape(-1, args.dim)[:T]

    # --- fish torch layers on the same input (fp32), interleaved RoPE (fish)
    x = fish_x[:T].view(1, T, -1).float()
    input_pos = torch.arange(T)
    fc = setup["freqs_cis"][input_pos]
    # fish blocks attend over the FULL kv cache (max_seq_len keys), so the mask is (1,1,T,max_seq_len)
    m_full = setup["causal"][None, None, input_pos, : setup["fcfg"].max_seq_len]
    m = setup["causal"][None, None, input_pos, :T]
    with torch.inference_mode():
        h = x
        for blk in setup["layers"]:
            h = blk(h, fc, m_full, input_pos=input_pos)
    p_layers = pcc(h_tt, h[0])
    print(f"[seq {seq}] TT {N_LAYERS} layer(s) vs fish torch (interleaved RoPE): PCC {p_layers:.6f}")
    # also per-position PCC for the first/last positions to spot position-dependent (RoPE) errors
    print("   per-position PCC:", [round(pcc(h_tt[i], h[0, i]), 4) for i in (0, 1, T // 2, T - 1)])

    # --- rotate-half variant of layer 0 attention to identify a convention mismatch
    blk = setup["layers"][0]
    fcfg = setup["fcfg"]
    with torch.inference_mode():
        hn = blk.attention_norm(x)
        qkv = blk.attention.wqkv(hn)
        q, k, v = qkv.split(
            [fcfg.n_head * fcfg.head_dim, fcfg.n_local_heads * fcfg.head_dim, fcfg.n_local_heads * fcfg.head_dim],
            dim=-1,
        )
        q = q.view(1, T, fcfg.n_head, fcfg.head_dim)
        k = k.view(1, T, fcfg.n_local_heads, fcfg.head_dim)
        v = v.view(1, T, fcfg.n_local_heads, fcfg.head_dim)
        if getattr(blk.attention, "q_norm", None) is not None:
            q = blk.attention.q_norm(q)
            k = blk.attention.k_norm(k)
        q2, k2 = _rotate_half_variant(q, fc), _rotate_half_variant(k, fc)
        q2, k2, v2 = (t.transpose(1, 2) for t in (q2, k2, v))
        k2 = k2.repeat_interleave(fcfg.n_head // fcfg.n_local_heads, dim=1)
        v2 = v2.repeat_interleave(fcfg.n_head // fcfg.n_local_heads, dim=1)
        y = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2, attn_mask=m)
        y = y.transpose(1, 2).reshape(1, T, -1)
        h_rh = x + blk.attention.wo(y)
        h_rh = h_rh + blk.feed_forward(blk.ffn_norm(h_rh))
    if N_LAYERS == 1:
        print(f"[seq {seq}] TT layer vs fish torch with ROTATE-HALF RoPE: PCC {pcc(h_tt, h_rh[0]):.6f}")

    # --- norm + LM head tail on the last tile
    out_full = model.ttnn_prefill_forward(
        inputs[0],
        rot_mats_global=inputs[1],
        rot_mats_local=inputs[2],
        user_id=0,
        page_table=None,
        get_last_token=((T - 1) // 32) * 32,
    )
    out_host = out_full.cpu() if hasattr(out_full, "cpu") else ttnn.from_device(out_full)
    logits_tt = model.process_output_prefill(out_host, (T - 1) % 32).float().reshape(-1)[: cfg.slow.vocab_size]
    hid_tt = model.read_last_hidden()[(T - 1) % 32]
    with torch.inference_mode():
        normw = setup["slow"]["norm.weight"].float()
        hl = h[0, T - 1]
        hn_ref = hl * torch.rsqrt((hl * hl).mean() + fcfg.norm_eps) * normw
        logits_ref = E @ hn_ref
    print(
        f"[seq {seq}] post-norm hidden (tap) vs ref: PCC {pcc(hid_tt, hn_ref):.6f}; logits vs ref: PCC {pcc(logits_tt, logits_ref):.6f}; argmax {int(logits_tt.argmax())} vs {int(logits_ref.argmax())}"
    )
    assert p_embed > 0.999
    assert p_layers > 0.98, f"layer PCC {p_layers}"


def test_generator_path_vs_fish(setup, snapshot):
    """The exact S2Generator plumbing (tt_transformers Generator prefill + traced decode + hidden tap) against
    the fish torch layer stack, on a real prompt; localises a plumbing bug the direct-call test cannot see."""
    from models.tt_transformers.tt.generator import Generator

    cfg, model, args = setup["cfg"], setup["model"], setup["args"]
    E = setup["slow"]["tok_embeddings.weight"].float()
    CB = setup["cb"].float()
    normw = setup["slow"]["norm.weight"].float()
    fcfg = setup["fcfg"]
    tok = S2Tokenizer(snapshot)
    prompt = build_prompt(tok, "The quick brown fox jumps over the lazy dog.")
    T = prompt.shape[1]
    gen = Generator([model], [args], model.mesh_device)

    def fish_embed(tokens, codes):
        x = E[tokens]
        vq = sum(CB[codes[i] + i * cfg.codebook_size] for i in range(cfg.num_codebooks))
        m = (tokens >= cfg.semantic_begin_id) & (tokens <= cfg.semantic_end_id)
        return torch.where(m.view(-1, 1), (x + vq) / math.sqrt(cfg.num_codebooks + 1), x)

    def fish_tail(h):
        hn = h * torch.rsqrt((h * h).mean() + fcfg.norm_eps) * normw
        return hn, E @ hn

    # reset fish kv caches and run the prefill reference
    for blk in setup["layers"]:
        blk.attention.kv_cache = FR.KVCache(1, 4096, fcfg.n_local_heads, fcfg.head_dim, dtype=torch.float32)
    x = fish_embed(prompt[0], prompt[1:]).view(1, T, -1)
    pos = torch.arange(T)
    fc = setup["freqs_cis"][pos]
    mfull = setup["causal"][None, None, pos, : fcfg.max_seq_len]
    with torch.inference_mode():
        h = x
        for blk in setup["layers"]:
            h = blk(h, fc, mfull, input_pos=pos)
        hn_ref, logits_ref = fish_tail(h[0, T - 1])

    # TT via Generator (what S2Generator does)
    model.set_frame_codes(prompt[1:], offset=0)
    out = gen.prefill_forward_text(
        prompt[0].view(1, T), page_table=None, kv_cache=None, prompt_lens=torch.tensor([T]), enable_trace=False
    )
    print(f"\n[gen] prefill_forward_text returned {type(out).__name__} shape {tuple(getattr(out, 'shape', ()))}")
    logits_tt = out.reshape(-1)[: cfg.slow.vocab_size].float()
    hid = model.read_last_hidden()
    print(f"[gen] read_last_hidden shape {tuple(hid.shape)}")
    hid_tt = hid[(T - 1) % hid.shape[0]]
    p_l, p_h = pcc(logits_tt, logits_ref), pcc(hid_tt, hn_ref)
    print(
        f"[gen] PREFILL logits PCC {p_l:.5f} (argmax {int(logits_tt.argmax())} vs {int(logits_ref.argmax())}); hidden PCC {p_h:.5f}"
    )
    # which row of the tap tile matches best?
    rows = [round(pcc(hid[r], hn_ref), 3) for r in range(hid.shape[0])]
    print(
        f"[gen] hidden PCC per tap row: best row {int(torch.tensor(rows).argmax())} = {max(rows)} (expected row {(T-1) % 32})"
    )

    # one teacher-forced decode step: feed a plausible next frame
    g = torch.Generator().manual_seed(1)
    frame = torch.cat(
        [
            torch.tensor([int(logits_ref.argmax())]),
            torch.tensor([int(logits_ref.argmax()) - cfg.semantic_begin_id]),
            torch.randint(0, 1024, (9,), generator=g),
        ]
    )
    xd = fish_embed(frame[:1], frame[1:].view(-1, 1)).view(1, 1, -1)
    posd = torch.tensor([T])
    fcd = setup["freqs_cis"][posd]
    md = setup["causal"][None, None, posd, : fcfg.max_seq_len]
    with torch.inference_mode():
        hd = xd
        for blk in setup["layers"]:
            hd = blk(hd, fcd, md, input_pos=posd)
        hn_ref_d, logits_ref_d = fish_tail(hd[0, 0])
    model.set_frame_codes(frame[1:].view(-1, 1), offset=T)
    for step in range(2):  # first call captures the trace, second replays it
        outd = gen.decode_forward(
            frame[:1].view(1, 1),
            torch.tensor([T]),
            page_table=None,
            kv_cache=None,
            enable_trace=True,
            read_from_device=True,
        )
        ld = (outd[0] if isinstance(outd, tuple) else outd).reshape(-1)[: cfg.slow.vocab_size].float()
        hdt = model.read_last_hidden()[0]
        print(
            f"[gen] DECODE step (call {step}) logits PCC {pcc(ld, logits_ref_d):.5f} (argmax {int(ld.argmax())} vs {int(logits_ref_d.argmax())}); hidden PCC {pcc(hdt, hn_ref_d):.5f}"
        )
    assert p_l > 0.98 and p_h > 0.98
    assert pcc(ld, logits_ref_d) > 0.98
