# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-9B prefill on one (1, 4) Blackhole column (Qwen36Model.from_pretrained(prefill_profile="column")).

The requests are prompt-upsampler requests (the FLUX.3 Klein prompt upsampler is a Qwen3.5-9B VLM; text or one
reference image): the system turn is prefilled once (the prefix), then each request's tail is prefilled from that state
as one grouped-prefill row (Bg = 1) in a traced bucket, as the serving runner does. Image rows are the HF vision tower's
output (the vision tower is not part of this path). Reference: HF Qwen3_5ForConditionalGeneration (bf16, host)
last-row logits, cached in the model's weight cache directory.

* test_prefill_column_accuracy  t2i @128, 1 x 512^2 @384, 1 MP @1152: last-row logits PCC >= 0.998, top-1 == HF.
* test_prefill_column_perf      the 1 MP request @1152: trace replay + logits read, median of 7 isolated runs
                                <= PERF_TEXT_MS (vision tower and host staging not included).
* test_column_profile_*         host only: the profile fields and its AGMM table.

Run (HF_MODEL = the upsampler checkpoint or Qwen/Qwen3.5-9B):
  MESH_DEVICE=P150x4 pytest -svq models/demos/blackhole/qwen36/tests/test_prefill_column.py
  Galaxy column: add TT_VISIBLE_DEVICES=0,4,24,28 TT_MESH_GRAPH_DESC_PATH=<(1, 4) ring MGD> QWEN36_FABRIC_RING=1
"""

import json
import statistics
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp

BLOCK = 64
PREFIX_ALIGN = 128
PERF_TEXT_MS = 62.0  # Galaxy column: trace ~49 ms + the logits read over x1 PCIe
_REPO = Path(__file__).resolve().parents[5]
_IMAGES = (
    "models/demos/multimodal/gemma3/dog.jpg",
    "models/sample_data/ILSVRC2012_val_00048736.JPEG",
    "models/demos/stable_diffusion_xl_base/reference/output/sdxl_input_1024x1024.jpg",
)
_SCAFFOLD = (
    "Reference images follow in order (the first image, the second image, ...).\nUser request: {}\nOutput the caption."
)
_SYSTEM_FALLBACK = (
    "You expand short image requests into one detailed, literal caption for an image generator. Describe subjects, "
    "layout, lighting, materials and style. Output only the caption."
)
# name -> (prompt, image count, image size, bucket)
REQUESTS = {
    "t2i": ("a red fox in a snowy birch forest at dawn, soft golden light, mist between the trunks", 0, 0, 128),
    "img1": ("make it look like a watercolor painting", 1, 512, 384),
    "img1mp": ("make it look like a watercolor painting", 1, 1024, 1152),
}


def _images(n, size):
    from PIL import Image, ImageOps

    srcs = [Image.open(_REPO / f).convert("RGB") for f in _IMAGES]
    out = []
    for i in range(n):
        im, v = srcs[i % len(srcs)], i // len(srcs)
        w, h = im.size
        s = min(w, h) if v == 0 else int(min(w, h) * (0.85 - 0.1 * (v % 3)))
        x0 = ((w - s) // 2, 0, w - s, (w - s) // 3)[v % 4]
        y0 = ((h - s) // 2, h - s, 0, (h - s) // 2)[v % 4]
        c = im.crop((x0, y0, x0 + s, y0 + s)).resize((size, size), Image.BICUBIC)
        out.append(ImageOps.mirror(c) if v % 2 else c)
    return out


def build_request(ckpt_dir, name):
    """input_ids [1, L], pixel_values / image_grid_thw (or None) and n_sys, the length of the system turn every request
    of its task shares (the prefix)."""
    from transformers import AutoProcessor

    prompt, n_img, size, _ = REQUESTS[name]
    proc = AutoProcessor.from_pretrained(ckpt_dir)
    sys_path = Path(ckpt_dir) / "system_prompts.json"
    key = "i2i" if n_img else "t2i"
    system = json.loads(sys_path.read_text())[key] if sys_path.exists() else _SYSTEM_FALLBACK
    images = _images(n_img, size)
    content = (
        [{"type": "image"} for _ in images] + [{"type": "text", "text": _SCAFFOLD.format(prompt)}] if images else prompt
    )
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = proc(text=text, images=images or None, return_tensors="pt")
    sys_text = "<|im_start|>system\n" + system + "<|im_end|>\n"
    n_sys = len(proc.tokenizer.encode(sys_text, add_special_tokens=False))
    return dict(enc), n_sys


def hf_reference(ckpt_dir, cache_dir, name):
    """{"enc", "n_sys", "logits" (last row, fp32), "vision" (merged image rows, fp32, or None)}; cached."""
    path = Path(cache_dir) / "prefill_column_goldens" / f"{name}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    from transformers import Qwen3_5ForConditionalGeneration

    enc, n_sys = build_request(ckpt_dir, name)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(ckpt_dir, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        vision = None
        if enc.get("pixel_values") is not None:
            v = model.model.visual(enc["pixel_values"].to(torch.bfloat16), grid_thw=enc["image_grid_thw"])
            vision = (v.pooler_output if hasattr(v, "pooler_output") else v[0]).float()
        logits = model(**enc, logits_to_keep=1).logits[0, -1].float()
    ref = {"enc": enc, "n_sys": n_sys, "logits": logits, "vision": vision}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ref, path)
    return ref


class ColumnPrefill:
    """One request at a time: prefix (system turn) prefilled once, the tail as a traced Bg = 1 grouped-prefill row."""

    def __init__(self, model, mesh):
        self.model, self.mesh, self.a = model, mesh, model.args
        self.rep = ttnn.ReplicateTensorToMesh(mesh)
        self.nb = 256  # blocks: [0, 16) t2i prefix, [16, 32) i2i prefix, [64, 128) the row, the last one = trash
        model.prefill_trash_block = self.nb - 1
        model.allocate_kv_caches(
            [self.nb, self.a.n_local_kv_heads, BLOCK, self.a.head_dim], ttnn.bfloat16, batch_size=1
        )
        self.pt = torch.arange(self.nb, dtype=torch.int32).reshape(1, -1)
        self.gdn = [l.attention for l in model.layers if not l.is_full_attention]
        self.cfgs, self.traces = {}, {}
        self.init = {
            id(dn): {
                "rec": self._dev(torch.zeros(1, dn.Nv, dn.Dk, dn.Dv), ttnn.float32),
                "pad": self._dev(torch.zeros(1, 32, dn.qkv_dim_tp), ttnn.bfloat16),
                "hist": self._dev(torch.zeros(1, 3, dn.qkv_dim_tp), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            }
            for dn in self.gdn
        }
        # the prefix state of each task (system turn), copied into self.init before a request of that task
        self.snap = {
            task: {id(dn): {k: ttnn.clone(v) for k, v in self.init[id(dn)].items()} for dn in self.gdn}
            for task in ("t2i", "i2i")
        }
        model._ensure_gdn_prefill_scratch()
        self.warm_pt = self.pt[:, :64]
        model.prepare_prefill_trace_chunked(mesh, self.warm_pt, chunk_size=2048, warmup_masked_buckets=True)
        model.prepare_prefill_traces_short(mesh, self.warm_pt, (128, 256, 512, 1024))

    def _dev(self, t, dtype, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=self.mesh, mesh_mapper=self.rep)

    def _row_table(self, task, P):
        """The task's prefix blocks, then the row's own blocks for [P, ...)."""
        t = torch.zeros(1, 64, dtype=torch.int32)
        pb, base = P // BLOCK, {"t2i": 0, "i2i": 16}[task]
        assert pb <= 16, P
        t[0, :pb] = torch.arange(base, base + pb, dtype=torch.int32)
        t[0, pb:] = torch.arange(64, 128 - pb, dtype=torch.int32)
        return t

    def add_bucket(self, Tb):
        """Persistent buffers of one bucket (all allocated before any trace is captured)."""
        m, a = self.model, self.a
        cfg = m._alloc_batched_short_cfg(Tb, 1, 64)
        rd = int(cfg.cos.shape[-1])
        cfg.cos_rows, cfg.sin_rows = [ttnn.clone(cfg.cos)], [ttnn.clone(cfg.sin)]
        cfg.cos_cat = self._dev(torch.ones(1, 1, Tb, rd), ttnn.bfloat16)
        cfg.sin_cat = self._dev(torch.zeros(1, 1, Tb, rd), ttnn.bfloat16)
        shard = ttnn.ShardTensor2dMesh(self.mesh, dims=(None, -1), mesh_shape=a.cluster_shape)
        cfg.vis = ttnn.from_torch(
            torch.zeros(1, 1, Tb, a.dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=shard,
        )
        cfg.vmask = self._dev(torch.zeros(1, 1, Tb, 1), ttnn.bfloat16)
        self.cfgs[Tb] = cfg

    def body(self, Tb, conv_sel_tail):
        m, cfg = self.model, self.cfgs[Tb]
        x = m.embd(cfg.tok)
        x = ttnn.to_memory_config(ttnn.reshape(x, (1, 1, Tb, x.shape[-1])), ttnn.DRAM_MEMORY_CONFIG)
        x2 = ttnn.where(cfg.vmask, cfg.vis, x)
        ttnn.deallocate(x)
        x = x2
        pg = dict(
            Bg=1,
            Tb=Tb,
            cos=cfg.cos_rows,
            sin=cfg.sin_rows,
            csi=cfg.csi,
            full_pts=cfg.full_pts,
            chunk_pts=cfg.chunk_pts,
            gdn_masks=cfg.gdn_masks,
            gdn_scratch=m._batched_gdn_scratch[1],
            gdn_init=self.init,
            cos_cat=cfg.cos_cat,
            sin_cat=cfg.sin_cat,
            conv_sel_tail=conv_sel_tail,
        )
        for layer in m.layers:
            xn = layer.forward(x, cos=cfg.cos_rows[0], sin=cfg.sin_rows[0], mode="prefill", prefill_group=pg)
            ttnn.deallocate(x)
            x = xn
        hidden = ttnn.reshape(x, (1, 1, Tb, x.shape[-1]))
        sel = cfg.sel if cfg.sel.dtype == hidden.dtype else ttnn.typecast(cfg.sel, hidden.dtype)
        x_last = ttnn.to_memory_config(ttnn.matmul(sel, hidden), ttnn.DRAM_MEMORY_CONFIG)
        from models.tt_transformers.tt.common import Mode

        x_last = m.norm(x_last, mode=Mode.PREFILL)
        from models.demos.blackhole.qwen36.tt.tp_common import lm_head_kwargs

        return ttnn.linear(x_last, m.lm_head_weight, **lm_head_kwargs(self.a))

    def prefill_prefix(self, task, ids, P):
        """The system turn [0, P) through the B = 1 prefill, its GDN state kept as the task's snapshot."""
        m = self.model
        toks = ids[:, :P].to(torch.int32)
        m._build_request_rope(toks, None)
        m._prefill_range(toks, self._row_table(task, P), 0, P, reset_gdn=True, want_logits=False)
        for dn in self.gdn:
            st = self.snap[task][id(dn)]
            rec = dn.rec_state if dn.rec_state.dtype == ttnn.float32 else ttnn.typecast(dn.rec_state, ttnn.float32)
            ttnn.copy(rec, st["rec"])
            if rec is not dn.rec_state:
                ttnn.deallocate(rec)
            hc = [ttnn.to_torch(t).to(torch.bfloat16) for t in ttnn.get_device_tensors(dn.conv_carry)]  # [1, K-1, C]
            hist = torch.stack([h[0, -3:] for h in hc])
            shard = ttnn.ShardTensorToMesh(self.mesh, dim=0)
            h = ttnn.from_torch(hist, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=shard)
            ttnn.copy_host_to_device_tensor(h, st["hist"])
        ttnn.synchronize_device(self.mesh)

    def load_prefix(self, task):
        for dn in self.gdn:
            for k, v in self.snap[task][id(dn)].items():
                ttnn.copy(v, self.init[id(dn)][k])

    def stage(self, Tb, task, ref, P):
        """Host staging of one request's tail [P, L): tokens, page tables, rope rows, image rows, masks."""
        m, a, cfg = self.model, self.a, self.cfgs[Tb]
        ids = ref["enc"]["input_ids"]
        n = ids.shape[1] - P
        assert 3 <= n <= Tb, (n, Tb)
        tok = torch.zeros(1, Tb, dtype=torch.int32)
        tok[0, :n] = ids[0, P:].to(torch.int32)
        row = self._row_table(task, P)
        nreal, cb0 = -(-ids.shape[1] // BLOCK), P // BLOCK
        row[0, nreal : cb0 + Tb // BLOCK] = m.prefill_trash_block
        grid = ref["enc"].get("image_grid_thw")
        m.rope.build_request_rope(ids, image_grid_thw=grid, video_grid_thw=None)
        cos, sin = m._rope_tp_cos_sin_torch(P, Tb)
        vis = torch.zeros(1, 1, Tb, a.dim)
        vmask = torch.zeros(1, 1, Tb, 1)
        if ref["vision"] is not None:
            pos = torch.nonzero(ids[0, P:] == m._vision_placeholder_token_id()).reshape(-1)
            vis[0, 0, pos] = ref["vision"].reshape(-1, a.dim)[: pos.numel()]
            vmask[0, 0, pos, 0] = 1.0
        bg, conv_sel, sel = m._batched_host_masks_torch(Tb, [n], cfg.n_hist)
        shard = ttnn.ShardTensor2dMesh(self.mesh, dims=(None, -1), mesh_shape=a.cluster_shape)
        for t, dt, lay, dst, mapper in (
            (tok, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, cfg.tok, self.rep),
            (row[:, :64].contiguous(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, cfg.full_pts[0], self.rep),
            (
                row[:, cb0 : cb0 + Tb // BLOCK].contiguous(),
                ttnn.int32,
                ttnn.ROW_MAJOR_LAYOUT,
                cfg.chunk_pts[0],
                self.rep,
            ),
            (cos, ttnn.bfloat16, ttnn.TILE_LAYOUT, cfg.cos_cat, self.rep),
            (sin, ttnn.bfloat16, ttnn.TILE_LAYOUT, cfg.sin_cat, self.rep),
            (bg, ttnn.float32, ttnn.TILE_LAYOUT, cfg.m_bg, self.rep),
            (conv_sel, ttnn.bfloat16, ttnn.TILE_LAYOUT, cfg.conv_sel, self.rep),
            (sel, ttnn.bfloat16, ttnn.TILE_LAYOUT, cfg.sel, self.rep),
            (torch.tensor([P], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, cfg.csi, self.rep),
            (vmask, ttnn.bfloat16, ttnn.TILE_LAYOUT, cfg.vmask, self.rep),
            (vis, ttnn.bfloat16, ttnn.TILE_LAYOUT, cfg.vis, shard),
        ):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(t, dtype=dt, layout=lay, mesh_mapper=mapper), dst)

    def warm(self, Tb, conv_sel_tail):
        for _ in range(2):  # compile; every allocation of the body happens here, before any group trace exists
            ttnn.deallocate(self.body(Tb, conv_sel_tail))
        ttnn.synchronize_device(self.mesh)

    def capture(self, Tb, conv_sel_tail):
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        out = self.body(Tb, conv_sel_tail)
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        self.traces[Tb] = (tid, out)

    def run(self, Tb):
        """Replay the bucket's trace; host logits [vocab] of the last real row."""
        tid, out = self.traces[Tb]
        ttnn.execute_trace(self.mesh, tid, cq_id=0, blocking=False)
        lg = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1))
        return lg.reshape(-1)[: self.a.vocab_size].float()


def _task(name):
    return "i2i" if REQUESTS[name][1] else "t2i"


def _conv_sel_tail(n_tail, Tb):
    """The smallest tile multiple W with n_tail - 3 >= Tb - W (the conv-state select reads only the last W rows)."""
    return min(Tb, -(-(Tb - (n_tail - 3)) // 32) * 32)


def _setup(mesh_device, names):
    """Model, HF references, prefix snapshots of both tasks and one captured trace per request (all allocations first:
    buffers, eager warm-up of every body, prefix prefill, then the captures)."""
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    model = Qwen36Model.from_pretrained(
        mesh_device, max_batch_size=1, max_seq_len=8192, enable_mtp=False, prefill_profile="column"
    )
    refs = {n: hf_reference(model.args.CKPT_DIR, model.args.weight_cache_path(), n) for n in names}
    P = {n: (r["n_sys"] // PREFIX_ALIGN) * PREFIX_ALIGN for n, r in refs.items()}
    tail = {n: refs[n]["enc"]["input_ids"].shape[1] - P[n] for n in names}
    cp = ColumnPrefill(model, mesh_device)
    for Tb in sorted({REQUESTS[n][3] for n in names}):
        cp.add_bucket(Tb)
    for n in names:
        Tb = REQUESTS[n][3]
        cp.stage(Tb, _task(n), refs[n], P[n])
        cp.warm(Tb, _conv_sel_tail(tail[n], Tb))
    model.capture_prefill_traces_short(mesh_device, cp.warm_pt, buckets=(128, 256, 512, 1024))
    for n in names:
        cp.prefill_prefix(_task(n), refs[n]["enc"]["input_ids"], P[n])
    for n in names:
        Tb = REQUESTS[n][3]
        cp.capture(Tb, _conv_sel_tail(tail[n], Tb))
    return model, cp, refs, P, tail


def _check(cp, refs, P, tail, names):
    """Prefill each request's tail from its task's prefix state and compare with HF."""
    out = {}
    for n in names:
        Tb = REQUESTS[n][3]
        cp.load_prefix(_task(n))
        cp.stage(Tb, _task(n), refs[n], P[n])
        lg = cp.run(Tb)
        ref = refs[n]["logits"]
        pcc = float(comp_pcc(lg, ref, 0.0)[1])
        top1 = int(lg.argmax()) == int(ref.argmax())
        logger.info(f"{n}: prefix {P[n]}, tail {tail[n]} @{Tb}: logits pcc {pcc:.5f}, top1 {'=' if top1 else '!='} HF")
        out[n] = (pcc, top1)
    return out


@torch.no_grad()
@parametrize_mesh_tp(max_tp=4)
def test_prefill_column_accuracy(mesh_device):
    assert tuple(mesh_device.shape) == (1, 4), "the column profile is measured on a (1, 4) mesh"
    names = list(REQUESTS)
    _, cp, refs, P, tail = _setup(mesh_device, names)
    res = _check(cp, refs, P, tail, names)
    bad = {n: r for n, r in res.items() if r[0] < 0.998 or not r[1]}
    assert not bad, bad


@torch.no_grad()
@parametrize_mesh_tp(max_tp=4)
def test_prefill_column_perf(mesh_device):
    """The longest request type (one 1 MP image + prompt, bucket 1152): trace replay + logits read."""
    assert tuple(mesh_device.shape) == (1, 4)
    name, Tb = "img1mp", 1152
    _, cp, refs, P, tail = _setup(mesh_device, [name])
    res = _check(cp, refs, P, tail, [name])
    assert res[name][1]
    ts = []
    for _ in range(7):
        ttnn.synchronize_device(mesh_device)
        time.sleep(2)  # isolated requests (the clock drops under sustained load)
        t0 = time.perf_counter()
        cp.run(Tb)
        ts.append((time.perf_counter() - t0) * 1e3)
    med = statistics.median(ts)
    logger.info(f"{name}: text prefill median {med:.1f} ms (runs {', '.join(f'{t:.1f}' for t in ts)})")
    assert med <= PERF_TEXT_MS, (med, PERF_TEXT_MS)


def _bare_args(dim=4096, experts=0, cluster=(1, 4)):
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs, Qwen36PrefillOptions

    a = Qwen36ModelArgs.__new__(Qwen36ModelArgs)
    a.dim, a.moe_num_experts, a.cluster_shape = dim, experts, cluster
    a.prefill_options = Qwen36PrefillOptions()
    return a


def test_column_profile_fields(expect_error):
    from models.demos.blackhole.qwen36.tt import tp_common as tpc

    a = _bare_args()
    stock = a.prefill_options
    assert not (stock.exact_norms or stock.gdn_l1) and stock.lm_head_cfg is None
    assert tpc.norm_compute_cfg(a) is None
    a.set_column_prefill_profile()
    o = a.prefill_options
    assert o.exact_norms and o.gdn_l1 and o.lm_head_cfg == "hifi4_fp32"
    assert a.prefill_fused_dist_norm and a.prefill_agmm_bf8 and a.prefill_small_m_max == 128
    assert a.mlp_math_fidelity == ttnn.MathFidelity.HiFi2
    assert a.prefill_sdpa_config_fn(1152) == (128, 128, True)
    assert tpc.norm_compute_cfg(a).math_fidelity == ttnn.MathFidelity.HiFi4
    with expect_error(AssertionError, "dense dim-4096"):
        _bare_args(experts=8).set_column_prefill_profile()


def test_column_profile_agmm_table():
    a = _bare_args()
    a.set_column_prefill_profile()
    f = a.prefill_agmm_config_fn
    assert f(128, "gdn_in") == {"grid_x": 8, "grid_y": 9, "num_workers_per_link": 6, "M_block": 4, "N_block": 8}
    assert f(1152, "gdn_in")["M_block"] == 8  # site default
    assert f(1152, "gdn_out")["N_block"] == 16
    assert f(512, "gdn_out") is None  # qwen36 default
    assert f(2688, "gdn_in") is None  # measured below 2048 rows


def test_conv_sel_tail():
    assert _conv_sel_tail(1112, 1152) == 64
    assert _conv_sel_tail(109, 128) == 32
    assert _conv_sel_tail(3, 128) == 128
