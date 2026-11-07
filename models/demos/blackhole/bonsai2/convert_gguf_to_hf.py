# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Convert PrismML Ternary-Bonsai-2-27B (PQ2_0 GGUF, Hadamard-folded) to a dense
HF Qwen3.8-27B-layout bf16 safetensors checkpoint that models/demos/blackhole/qwen36 can load.

The GGUF stores W' = ternary(W @ (H D)^T) per weight, where the runtime transforms activations
x' = H (D x) blockwise (normalized Sylvester WHT, block 1024, explicit +-1 signs D).  Since H D is
orthogonal, W_dense = W' @ H @ D reproduces y = W' x' exactly.  Embedding is the inverse
manifest: e = D H e'  ->  E_dense = E' @ H @ D as well (H symmetric).

Usage:
  python models/demos/blackhole/bonsai2/convert_gguf_to_hf.py GGUF BASE_HF_DIR OUT_DIR [--validate-only]
BASE_HF_DIR (Qwen3.8-27B snapshot) supplies config/tokenizer and is used to sanity-check
orientation/permutation of every tensor (cosine similarity vs base weights).
"""

import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import torch

GGUF_TYPES = {  # type id -> (block elems, block bytes)
    0: (1, 4),  # F32
    1: (1, 2),  # F16
    30: (1, 2),  # BF16
    142: (128, 34),  # PQ2_0
    143: (128, 28),  # PTQ1_0
}


def read_gguf(path):
    f = open(path, "rb")
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    assert f.read(4) == b"GGUF"
    (version,) = struct.unpack("<I", f.read(4))
    n_t, n_kv = struct.unpack("<QQ", f.read(16))

    def rstr():
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    scal = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    def rval(t):
        if t in scal:
            fmt = scal[t]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
        if t == 8:
            return rstr()
        if t == 9:
            (et,) = struct.unpack("<I", f.read(4))
            (n,) = struct.unpack("<Q", f.read(8))
            return [rval(et) for _ in range(n)]
        raise ValueError(t)

    kv = {}
    for _ in range(n_kv):
        k = rstr()
        (t,) = struct.unpack("<I", f.read(4))
        kv[k] = rval(t)
    infos = []
    for _ in range(n_t):
        name = rstr()
        (nd,) = struct.unpack("<I", f.read(4))
        dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
        (tt,) = struct.unpack("<I", f.read(4))
        (off,) = struct.unpack("<Q", f.read(8))
        infos.append((name, dims, tt, off))
    align = kv.get("general.alignment", 32)
    base = (f.tell() + align - 1) // align * align
    tensors = {}
    for name, dims, tt, off in infos:
        be, bb = GGUF_TYPES[tt]
        n = int(np.prod(dims))
        nbytes = n // be * bb
        tensors[name] = (dims, tt, mm[base + off : base + off + nbytes])
    return kv, tensors


def dequant(dims, tt, raw):
    shape = tuple(int(d) for d in dims[::-1])  # numpy order [rows, cols]
    if tt == 0:
        return torch.from_numpy(raw.view("<f4").reshape(shape).copy())
    if tt == 1:
        return torch.from_numpy(raw.view("<f2").astype(np.float32).reshape(shape))
    if tt == 30:
        return torch.from_numpy(raw.view("<u2").astype(np.uint32).__lshift__(16).view("<f4").reshape(shape).copy())
    if tt == 142:  # PQ2_0: fp16 d, 32 bytes of 2-bit codes, value=(q-1)*d
        blk = raw.reshape(-1, 34)
        d = blk[:, :2].copy().view("<f2").astype(np.float32)  # [nb,1]
        qs = blk[:, 2:]
        q = np.stack([(qs >> s) & 3 for s in (0, 2, 4, 6)], axis=-1).reshape(-1, 128)
        return torch.from_numpy(((q.astype(np.float32) - 1.0) * d).reshape(shape))
    if tt == 143:  # PTQ1_0 (same trit decode as Prism MLX codec)
        blk = raw.reshape(-1, 28)
        d = blk[:, 26:28].copy().view("<f2").astype(np.float32)
        pieces = []
        for lo, hi, count in [(0, 16, 5), (16, 24, 5), (24, 26, 4)]:
            packed = blk[:, lo:hi].astype(np.uint16)
            for trit in range(count):
                rem = (packed * (3**trit)) & 255
                pieces.append(((rem * 3) >> 8).astype(np.uint8))
        q = np.concatenate(pieces, axis=1)
        return torch.from_numpy(((q.astype(np.float32) - 1.0) * d).reshape(shape))
    raise ValueError(tt)


def hadamard(n):
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / np.sqrt(n)).float()


def unrotate(w, block, signs, H):
    # w: [out, in]; W_dense = W' @ blockdiag(H) @ diag(signs)
    o, i = w.shape
    w = (w.reshape(o, i // block, block) @ H).reshape(o, i)
    return w * signs[None, :]


MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
}
GLOBAL = {
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.language_model.norm.weight",
    "token_embd.weight": "model.language_model.embed_tokens.weight",
}


def hf_name(name):
    if name in GLOBAL:
        return GLOBAL[name], name
    _, layer, stem = name.split(".", 2)
    return f"model.language_model.layers.{layer}.{MAP[stem]}", stem


class Base:
    def __init__(self, d):
        from safetensors import safe_open

        self.d = Path(d)
        self.wm = json.load(open(self.d / "model.safetensors.index.json"))["weight_map"]
        self.handles = {}
        self.safe_open = safe_open

    def get(self, k):
        fn = self.wm[k]
        if fn not in self.handles:
            self.handles[fn] = self.safe_open(str(self.d / fn), "pt")
        return self.handles[fn].get_tensor(k).float()


def cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(a @ b / (a.norm() * b.norm() + 1e-30))


def main():
    gguf, base_dir, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    validate_only = "--validate-only" in sys.argv
    kv, tensors = read_gguf(gguf)
    pk = {k: v for k, v in kv.items() if k.startswith("prism.")}
    print({k: (v if not isinstance(v, list) or len(v) < 8 else f"list[{len(v)}]") for k, v in pk.items()})
    block = int(kv["prism.hadamard.block_size"])
    widths, vals = kv["prism.hadamard.sign_widths"], kv["prism.hadamard.sign_values"]
    signs, off = {}, 0
    for w in widths:
        signs[w] = torch.tensor(vals[off : off + w], dtype=torch.float32)
        off += w
    folded = set(kv["prism.hadamard.weight_names"]) | set(kv.get("prism.hadamard.inverse_weight_names", []))
    H = hadamard(block)
    nv, nk = int(kv["qwen35.ssm.time_step_rank"]), int(kv["qwen35.ssm.group_count"])
    hd = int(kv["qwen35.ssm.inner_size"]) // nv
    hk = int(kv["qwen35.ssm.state_size"])
    base = Base(base_dir)
    print(f"{len(tensors)} tensors, block={block}, nv={nv} nk={nk} hd={hd} hk={hk}, folded={len(folded)}")

    # V-head order: llama.cpp regroups HF's [nk, nv/nk] v-heads to [nv/nk, nk] ("tiled").
    def v_to_hf(unit):  # index array g s.t. hf = gguf[g]
        return np.arange(nv * unit).reshape(nv // nk, nk, unit).transpose(1, 0, 2).reshape(-1)

    out_sd, report, wm, total = {}, [], {}, [0]
    shard_bytes = 0
    if not validate_only:
        out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    def flush():
        nonlocal shard_bytes
        if not out_sd:
            return
        fn = f"model-{len(set(wm.values())) + 1:05d}.safetensors"
        save_file(out_sd, str(out / fn), metadata={"format": "pt"})
        for k, t in out_sd.items():
            wm[k] = fn
            total[0] += t.numel() * 2
        print("saved", fn, flush=True)
        out_sd.clear()
        shard_bytes = 0

    names = list(tensors)
    if validate_only:
        names = [n for n in names if not n.startswith("blk.") or n.split(".")[1] in ("0", "3", "62")]
    for name in names:
        dims, tt, raw = tensors[name]
        target, stem = hf_name(name)
        w = dequant(dims, tt, raw)
        if name in folded:
            w = unrotate(w, block, signs[w.shape[1]], H)
        ref = base.get(target)
        # Fixups from llama.cpp qwen35 converter conventions -> HF.
        cands = {"as_is": w}
        if stem == "ssm_a":
            cands = {"log(-a)": torch.log(-w)}
        if stem == "ssm_conv1d.weight":
            w = w.unsqueeze(1)
            cands = {"as_is": w}
        if stem.endswith("norm.weight") and stem != "ssm_norm.weight":
            cands["minus1"] = w - 1.0
        if nv != nk:
            qk = 2 * nk * hk
            if stem in ("attn_qkv.weight", "ssm_conv1d.weight"):
                cands["vperm_rows"] = torch.cat([w[:qk], w[qk:][v_to_hf(hd)]], 0)
            if stem == "attn_gate.weight":
                cands["vperm_rows"] = w[v_to_hf(hd)]
            if stem in ("ssm_alpha.weight", "ssm_beta.weight", "ssm_dt.bias"):
                cands["vperm_rows"] = w[v_to_hf(1)]
            if stem == "ssm_a":
                cands["vperm_rows"] = torch.log(-w)[v_to_hf(1)]
            if stem == "ssm_out.weight":
                cands["vperm_cols"] = w[:, v_to_hf(hd)]
        if any(c.shape != ref.shape for c in cands.values()):
            raise ValueError(f"{name}: shape {[tuple(c.shape) for c in cands.values()]} vs ref {tuple(ref.shape)}")
        scores = {k: cos(c, ref) for k, c in cands.items()}
        if stem.endswith("norm.weight") or stem in ("ssm_a", "ssm_dt.bias"):
            # cosine is insensitive to the +1 offset when values are large; use rel error.
            scores = {k: 1 - float((c - ref).norm() / (ref.norm() + 1e-6)) for k, c in cands.items()}
        best = max(scores, key=scores.get)
        report.append((name, tt, best, scores))
        print(f"{name:40s} t={tt:3d} {tuple(w.shape)} best={best} {scores}", flush=True)
        if not validate_only:
            out_sd[target] = cands[best].to(torch.bfloat16).contiguous()
            shard_bytes += out_sd[target].numel() * 2
            if shard_bytes > 4e9:
                flush()
    if validate_only:
        return
    flush()
    json.dump(
        {"metadata": {"total_size": total[0]}, "weight_map": wm},
        open(out / "model.safetensors.index.json", "w"),
        indent=1,
    )
    for f in [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ]:
        if (Path(base_dir) / f).exists():
            shutil.copy(Path(base_dir) / f, out / f)
    json.dump(
        [(n, t, b, s) for n, t, b, s in report], open(out / "bonsai_conversion_report.json", "w"), indent=0, default=str
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
