# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The resident 45-layer mesh prefill (tt/mesh_model.py) graded against the tt-blaze CPU golden bundle
(``GLM53F_GOLDEN_DIR``: ``vllm-<job>/`` token streams + ``taps/bf16/<job>/layer_L.safetensors`` +
``state/bf16/<job>/layer_L.safetensors``; tt-blaze G53F-0017), the disaggregated-prefill source of the blaze
48-stage (1, 2) ring (tt-blaze G53F-0080).

Per job, two prefills of user slot 0:

* teacher-forced: the golden's ``n_group_a_rows`` tokens (prompt + its greedy continuation) through all layers,
  layer by layer; per layer the output streams vs the golden ``layer_out`` rows (PCC over all rows and the worst row),
  for DSA layers the cached latent rows vs ``dsa_kv_latent``, for KDA layers the final recurrent / conv carries vs the
  golden end-of-run ``state``; the host head on the device's final streams vs the golden next tokens (teacher-forced
  top-1 match).
* prompt only (``n_prompt_tokens``): wall time (TTFT, cold + warm), the device head's first token vs the golden's
  first completion token, and the per-layer state the ring would be seeded from (KDA recurrent + conv carries, DSA
  latent rows, the indexer pool cache + raw rings) saved under ``GLM53F_PREFILL_OUT/<job>_prompt/``.

    GLM53_MESH_SHAPE=8,4 GLM53F_PREFILL_JOBS=ka,kaqa,long pytest models/demos/glm53_flash/tests/test_prefill_vs_blaze_golden.py -s
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file

import ttnn
from models.demos.glm53_flash.reference.checkpoint import (
    FINAL_NORM_KEY,
    LM_HEAD_KEY,
    default_snapshot_dir,
    load_tensors,
    shard_headers,
)
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mesh import mesh_device_params
from models.demos.glm53_flash.tt.mesh_model import GLM53MeshModel

GOLDEN = Path(os.environ.get("GLM53F_GOLDEN_DIR", "/mnt/tt-data/sdawle/glm53f/golden_sweep"))
OUT = Path(os.environ.get("GLM53F_PREFILL_OUT", "/mnt/tt-data/sdawle/glm53f/prefill/out"))
JOBS = [j for j in os.environ.get("GLM53F_PREFILL_JOBS", "ka,kaqa,long").split(",") if j]
MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
MAX_SEQ = int(os.environ.get("GLM53F_PREFILL_MAXSEQ", "8192"))
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "256"))
_DT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
EXPERT_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DTYPE", "bfp8")]
EXPERT_DOWN_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DOWN_DTYPE", "bfp8")]
_cache_env = os.environ.get("GLM53_MESH_CACHE", "/mnt/tt-data/sdawle/glm53f/prefill/mesh_cache")
CACHE_ROOT = None if _cache_env.lower() == "none" else Path(_cache_env)
SAVE_STATES = os.environ.get("GLM53F_PREFILL_SAVE", "1") == "1"
DUMP_LAYERS = set(os.environ.get("GLM53F_PREFILL_DUMP_LAYERS", "").split(","))


def _layers() -> list[int]:
    spec = os.environ.get("GLM53_MODEL_LAYERS", "0-44")
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in spec.split(",")]


def _meta(job: str) -> dict:
    return json.loads((GOLDEN / f"vllm-{job}" / "metadata.json").read_text())


def _read(path: Path, keys: list[str] | None = None) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt") as f:
        return {k: f.get_tensor(k) for k in (keys or f.keys()) if k in f.keys()}


def _tap(job: str, L: int, keys: list[str]) -> dict[str, torch.Tensor]:
    return _read(GOLDEN / "taps" / "bf16" / job / f"layer_{L}.safetensors", keys)


def _trace(job: str, name: str) -> torch.Tensor:
    """A full-row golden stream of ``vllm-<job>/`` (``decoder_input_layer_0`` / ``decoder_output_layer_L``): every row,
    unlike the per-layer taps, which are sparse past the first rows of a long job."""
    d = GOLDEN / f"vllm-{job}"
    s = json.loads((d / "index.json").read_text())["tensor_streams"][name]
    parts = []
    for ch in s["chunks"]:
        with safe_open(str(d / ch["path"]), framework="pt") as f:
            (key,) = list(f.keys())
            parts.append(f.get_tensor(key))
    return torch.cat(parts) if len(parts) > 1 else parts[0]


def _tap_positions(job: str, L: int) -> list[int]:
    """The positions the layer's tap rows hold (``tap_row_ranges``; all rows when absent)."""
    t = _read(GOLDEN / "taps" / "bf16" / job / f"layer_{L}.safetensors", ["tap_row_ranges"]).get("tap_row_ranges")
    return None if t is None else [r for a, b in t.tolist() for r in range(a, b)]


def _state(job: str, L: int) -> dict[str, torch.Tensor]:
    return _read(GOLDEN / "state" / "bf16" / job / f"layer_{L}.safetensors")


def _row_pcc_min(a: torch.Tensor, b: torch.Tensor) -> tuple[float, int]:
    worst, at = 1.0, -1
    for r in range(a.shape[0]):
        p = pcc(a[r], b[r])
        if p < worst:
            worst, at = p, r
    return worst, at


def _embed(model: GLM53MeshModel, ids: torch.Tensor) -> tuple[ttnn.Tensor, int]:
    """``GLM53MeshModel.prefill``'s embedding step: the row slabs of the padded prompt."""
    S = int(ids.numel())
    unit = 32 * model.mesh.rows
    T_pad = -(-S // unit) * unit
    host = model.embedding.tokens_to_host(ids, pad_to=T_pad)
    if model.mesh.rows > 1:
        ids_dev = model.mesh.upload(host, ttnn.uint32, rows=3, layout=ttnn.ROW_MAJOR_LAYOUT)
    else:
        ids_dev = ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=model.device)
    streams = model.embedding(ids_dev)
    ttnn.deallocate(ids_dev)
    return streams, T_pad


def _streams_host(model: GLM53MeshModel, streams: ttnn.Tensor, S: int) -> torch.Tensor:
    return model.mesh.to_host(streams, rows=2 if model.mesh.rows > 1 else None).float()[0, 0, :S]  # [S, 16384]


def _kda_state_host(model: GLM53MeshModel, dec) -> dict[str, torch.Tensor]:
    """Slot 0's KDA carries, heads gathered over the TP columns (row 0's replica)."""
    st = dec.kda_batched
    rec = model.mesh.to_host(st.recurrent, cols=1)[0:1].float()  # [1, 64, 128, 128] (heads column-sharded head-major)
    tiled = st.convolution.layout == ttnn.TILE_LAYOUT  # tap-major tiles [taps, B, W_c] vs ttKDA's [B, taps, W_c]
    conv = model.mesh.to_host(st.convolution, cols=2)
    if tiled:
        conv = conv.permute(1, 0, 2)
    conv = conv[0:1].float()  # [1, taps, cols * W_c], chip c's block = [q_c | k_c | v_c]
    cols, taps = model.mesh.cols, conv.shape[1]
    # -> the HF channel order [q | k | v] (all heads each), the golden conv_state's
    conv = conv.reshape(1, taps, cols, 3, -1).permute(0, 1, 3, 2, 4).reshape(1, taps, -1)
    return {"recurrent_state": rec.contiguous(), "conv_history": conv.contiguous().to(torch.bfloat16)}


def _replicated_rows_host(model: GLM53MeshModel, t: ttnn.Tensor, nb: int) -> torch.Tensor:
    """The first ``nb`` dim-0 rows of a mesh-REPLICATED tensor ``t`` on the host (tt-blaze G53F-0182).

    Default: slice them on device and read one chip's replica (one chip's PCIe read: ~1.5 s per GiB). With
    ``GLM53F_EXPORT_PARTITION=1``: pad ``nb`` up to a multiple of the chip count D, ``ttnn.mesh_partition`` the slice
    over dim 0 (a per-device slice, no fabric: chip i keeps rows [i nb/D, (i+1) nb/D)), untilize the parts on device and
    read all D in one mesh read, concatenated in the mesh's device order -- every chip supplies 1/D of the bytes
    (tests/test_export_bw.py: 1 GiB in 0.66 s vs 1.52 s, bitwise equal). Falls back to the default when the padded count
    exceeds the tensor."""
    nb = min(nb, int(t.shape[0]))
    rest = [int(t.shape[i]) for i in range(1, len(t.shape))]
    D = model.mesh.rows * model.mesh.cols
    if os.environ.get("GLM53F_EXPORT_PARTITION", "0") == "1" and D > 1:
        nbp = -(-nb // D) * D
        if nbp <= int(t.shape[0]):
            part = ttnn.slice(t, [0, 0, 0, 0], [nbp, *rest])
            p = ttnn.mesh_partition(part, 0)
            if part.buffer_address() != t.buffer_address():  # a full-size slice IS t (the live cache): never free it
                ttnn.deallocate(part)
            if p.layout == ttnn.TILE_LAYOUT:  # untilize on device: the host untilize was 0.69 of 1.52 s per GiB
                q = ttnn.to_layout(p, ttnn.ROW_MAJOR_LAYOUT)
                ttnn.deallocate(p)
                p = q
            out = ttnn.to_torch(p, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh.device, dim=0))
            ttnn.deallocate(p)
            return out[:nb]
    part = ttnn.slice(t, [0, 0, 0, 0], [nb, *rest])
    out = model.mesh.to_host(part)  # first replica
    if part.buffer_address() != t.buffer_address():  # a full-size slice IS t (the live cache): never free it
        ttnn.deallocate(part)
    return out


def _dsa_state_host(model: GLM53MeshModel, dec, S: int) -> dict[str, torch.Tensor]:
    """Slot 0's latent rows [S, 512] (identity page table), plus the indexer pool cache and raw rings."""
    c = dec.cache
    lat = _replicated_rows_host(model, c.cache, -(-S // 32)).reshape(-1, int(c.cache.shape[-1]))[:S]
    out = {"latent": lat.contiguous().to(torch.bfloat16)}
    idx = getattr(c, "index", None) or getattr(dec, "index", None)
    if idx is None:
        idx = next((v for v in vars(c).values() if type(v).__name__ == "TtIndexCache"), None)
    if idx is not None:
        # likewise only slot 0's blocks holding pools [0, ceil(S/4)), in the cache's own bf16 (the ring's dsa_history
        # widens to fp32 itself): the full 1 M allocation is 262 400 rows = 134 MB per layer as fp32 (G53F-0089)
        np_ = -(-S // 4)
        pb = min(-(-np_ // 32), int(idx.pool_blocks))
        pool = _replicated_rows_host(model, idx.pool_cache, pb)
        out["index_pool"] = pool.reshape(-1, pool.shape[-1])[:np_].contiguous().to(torch.bfloat16)
        out["index_ring_k"] = model.mesh.to_host(idx.ring_k)[0, 0].contiguous().float()
        out["index_ring_g"] = model.mesh.to_host(idx.ring_g)[0, 0].contiguous().float()
    return out


def _save_seed(model: GLM53MeshModel, streams, d: Path, job: str, toks: list[int], n: int, next_token: int) -> float:
    """Every layer's state after a prefill of ``toks[:n]`` (slot 0) + the final streams, one safetensors per layer.
    Returns the export wall time (device -> host -> file), seconds."""
    t0 = time.perf_counter()
    d.mkdir(parents=True, exist_ok=True)
    save_file({"streams": _streams_host(model, streams, n).contiguous()}, str(d / "final_streams.safetensors"))
    for dec in model.layers:
        st = _dsa_state_host(model, dec, n) if dec.cache is not None else _kda_state_host(model, dec)
        save_file(st, str(d / f"layer_{dec.layer_idx}.safetensors"))
    wall = time.perf_counter() - t0
    meta = {
        "job": job,
        "n": n,
        "tokens": toks[:n],
        "next_token": next_token,
        "layers": [x.layer_idx for x in model.layers],
        "export_s": round(wall, 3),
    }
    (d / "meta.json").write_text(json.dumps(meta))
    logger.info(f"[g53f-prefill] seed {d.name}: {len(model.layers)} layers exported in {wall:.2f} s")
    return wall


@pytest.mark.timeout(21600)
@pytest.mark.parametrize(
    "device_params", [mesh_device_params(*MESH, trace_region_size=TRACE_MB * 1_000_000)], indirect=True
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_prefill_vs_blaze_golden(mesh_device):
    layers = _layers()
    snapshot = default_snapshot_dir()
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    t0 = time.time()
    model = GLM53MeshModel(
        mesh_device,
        cfg,
        layers,
        max_batch=1,
        max_seq_len=MAX_SEQ,
        weight_cache_path=CACHE_ROOT,
        expert_dtype=EXPERT_DTYPE,
        expert_down_dtype=EXPERT_DOWN_DTYPE,
    )
    logger.info(f"[g53f-prefill] model {MESH} layers {layers[0]}..{layers[-1]} built in {time.time() - t0:.0f} s")
    w = load_tensors(snapshot, [FINAL_NORM_KEY, LM_HEAD_KEY], shard_headers(snapshot))
    norm_w, head_w = w[FINAL_NORM_KEY].float(), w[LM_HEAD_KEY].to(torch.bfloat16)
    summary: dict = {}
    from models.demos.glm53_flash.tt import mhc as mhc_mod

    # tt-blaze G53F-0187: GLM53F_PREFILL_MHC_MODES="wrap,fast" grades every job under each prefill mHC path in one process
    modes = [m for m in os.environ.get("GLM53F_PREFILL_MHC_MODES", "").split(",") if m] or [None]
    runs = [(job, m) for m in modes for job in JOBS]
    for job, mode in runs:
        if mode is not None:
            mhc_mod.MHC_PREFILL_RUNTIME = "" if mode == "wrap" else mode
            logger.info(
                f"[g53f-prefill] ===== prefill mHC mode {mode} (MHC_PREFILL_RUNTIME={mhc_mod.MHC_PREFILL_RUNTIME!r})"
            )
        meta = _meta(job)
        # the golden rows = prompt + its greedy completion (metadata "token_ids" is the prompt only)
        toks = list(meta["prompt_token_ids"]) + list(meta["completion_token_ids"])
        S, n_prompt = int(meta["n_group_a_rows"]), int(meta["n_prompt_tokens"])
        rep: dict = {"S": S, "n_prompt": n_prompt, "layers": {}}
        # ---- teacher-forced: every golden row through the layers, one layer at a time
        ids = torch.tensor(toks[:S], dtype=torch.int64)
        streams, _ = _embed(model, ids)
        emb = _streams_host(model, streams, S)
        g_in = (
            _trace(job, "decoder_input_layer_0")
            if layers[0] == 0
            else _trace(job, f"decoder_output_layer_{layers[0] - 1}")
        )
        rep["embed_pcc"] = round(pcc(g_in.float()[:S], emb), 6)
        logger.info(f"[g53f-prefill] {job} input streams vs golden layer_in of L{layers[0]}: pcc {rep['embed_pcc']}")
        t_job = time.time()
        for dec in model.layers:
            L = dec.layer_idx
            out = dec.prefill_forward(streams, user=0, seq_len=S, start=0)
            ttnn.deallocate(streams)
            streams = out
            dev = _streams_host(model, streams, S)
            gold = _trace(job, f"decoder_output_layer_{L}").float()[:S]
            p_all = pcc(gold, dev)
            p_min, at = _row_pcc_min(gold, dev)
            ent = {"layer_out_pcc": round(p_all, 6), "row_min": round(p_min, 6), "row_min_at": at}
            ent["stream_pcc"] = [
                round(pcc(gold[:, i * 4096 : (i + 1) * 4096], dev[:, i * 4096 : (i + 1) * 4096]), 4) for i in range(4)
            ]
            ent["norm_ratio"] = round(float(dev.norm() / gold.norm()), 4)
            if str(L) in DUMP_LAYERS:
                OUT.mkdir(parents=True, exist_ok=True)
                save_file(
                    {"dev": dev.contiguous(), "gold": gold.contiguous(), "emb": emb.contiguous()},
                    str(OUT / f"dump_{job}_L{L}.safetensors"),
                )
            if dec.cache is not None:  # DSA
                st = _dsa_state_host(model, dec, S)
                g = _tap(job, L, ["dsa_kv_latent"])["dsa_kv_latent"].float()
                rows = _tap_positions(job, L) or list(range(g.shape[0]))
                keep = [k for k, r in enumerate(rows) if r < S]
                ent["latent_pcc"] = round(pcc(g[keep], st["latent"].float()[[rows[k] for k in keep]]), 6)
            if getattr(dec, "kda_batched", None) is not None:  # KDA
                st = _kda_state_host(model, dec)
                gs = _state(job, L)
                ent["recurrent_pcc"] = round(pcc(gs["recurrent_state"].float(), st["recurrent_state"]), 6)
                # HF conv_state [1, W, 4] holds the last 4 conv inputs; the carry is the last 3 (taps-major)
                gc = gs["conv_state"].float()[0].t()[-3:]  # [3, W]
                ent["conv_pcc"] = round(pcc(gc, st["conv_history"].float()[0]), 6)
            rep["layers"][L] = ent
            logger.info(f"[g53f-prefill] {job} S={S} L{L}: {ent}")
        # host head on the device's final streams: rows n_prompt-1 .. S-1 predict tokens n_prompt .. S
        h = dev.reshape(S, 4, -1).mean(1)[n_prompt - 1 :]
        hn = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * norm_w
        pred = (hn.to(torch.bfloat16) @ head_w.t()).float().argmax(-1).tolist()
        want = toks[n_prompt : S + 1]
        match = sum(int(a == b) for a, b in zip(pred, want))
        rep["teacher_forced"] = {"match": match, "of": len(want), "pred": pred, "want": want}
        rep["teacher_forced_wall_s"] = round(time.time() - t_job, 1)
        logger.info(f"[g53f-prefill] {job} teacher-forced head top-1 {match}/{len(want)}: pred {pred} want {want}")
        ttnn.deallocate(streams)
        # ---- prompt only: TTFT + first token + the seed states
        prompt = torch.tensor(toks[:n_prompt], dtype=torch.int64)
        walls = []
        for it in range(2):
            ttnn.synchronize_device(mesh_device)
            t1 = time.perf_counter()
            st_out, logits, owner = model.prefill(prompt, user=0)
            tok = model.first_token(logits, owner)
            walls.append(time.perf_counter() - t1)
            ttnn.deallocate(logits)
            if it == 0:
                ttnn.deallocate(st_out)
        rep["prompt"] = {
            "first_token": tok,
            "golden_first": toks[n_prompt],
            "ttft_cold_s": walls[0],
            "ttft_warm_s": walls[1],
        }
        logger.info(
            f"[g53f-prefill] {job} prompt {n_prompt} tok: first token {tok} (golden {toks[n_prompt]}, "
            f"{'MATCH' if tok == toks[n_prompt] else 'DIFF'}); TTFT cold {walls[0]:.3f} s warm {walls[1]:.3f} s"
        )
        if SAVE_STATES:
            _save_seed(model, st_out, OUT / f"{job}_prompt", job, toks, n_prompt, tok)
        ttnn.deallocate(st_out)
        # ---- the ring seed: the prefill of ids[:S'] at the pool-aligned split S' = 4 * floor((n_prompt - 1) / 4), so the
        # k-pool ring holds no partial pool and the ring replays R = n_prompt - S' in [1, 4] tokens (its own first token)
        split = 4 * ((n_prompt - 1) // 4)
        if SAVE_STATES and split >= 4:
            st_out, logits, owner = model.prefill(torch.tensor(toks[:split], dtype=torch.int64), user=0)
            nxt = model.first_token(logits, owner)
            ttnn.deallocate(logits)
            _save_seed(model, st_out, OUT / f"{job}_split{split}", job, toks, split, nxt)
            rep["split"] = {"S_prime": split, "replay": n_prompt - split, "next_token": nxt, "golden_next": toks[split]}
            logger.info(
                f"[g53f-prefill] {job} seed at S'={split} (replay {n_prompt - split}): next {nxt} golden {toks[split]}"
            )
            ttnn.deallocate(st_out)
        # extra seeds of this job's rows (GLM53F_PREFILL_SEEDS="long:2044,..."): the prefill of toks[:n] at any 4-aligned n
        for spec in [s for s in os.environ.get("GLM53F_PREFILL_SEEDS", "").split(",") if s.startswith(f"{job}:")]:
            n = int(spec.split(":")[1])
            assert n % 4 == 0 and 4 <= n < len(
                toks
            ), f"GLM53F_PREFILL_SEEDS {spec}: n must be 4-aligned and < {len(toks)}"
            ttnn.synchronize_device(mesh_device)
            t1 = time.perf_counter()
            st_out, logits, owner = model.prefill(torch.tensor(toks[:n], dtype=torch.int64), user=0)
            nxt = model.first_token(logits, owner)
            wall = time.perf_counter() - t1
            ttnn.deallocate(logits)
            _save_seed(model, st_out, OUT / f"{job}_split{n}", job, toks, n, nxt)
            ttnn.deallocate(st_out)
            rep.setdefault("seeds", {})[n] = {"next_token": nxt, "golden_next": toks[n], "prefill_s": round(wall, 3)}
            logger.info(f"[g53f-prefill] {job} seed at S'={n}: next {nxt} golden {toks[n]} ({wall:.2f} s)")
        tag = job if mode is None else f"{job}_{mode}"
        rep["mhc_mode"] = mode
        summary[tag] = rep
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"report_{tag}.json").write_text(json.dumps(rep, indent=1))
    logger.info(
        f"[g53f-prefill] SUMMARY {json.dumps({j: {k: v for k, v in r.items() if k != 'layers'} for j, r in summary.items()})}"
    )
