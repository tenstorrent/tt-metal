# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A long video's audio track tokenized on the engine's row (MIMO_AUDIO_TOKENIZER=device, the default) against the
front end's fp32 CPU tokenizer encoder (MIMO_AUDIO_TOKENIZER=frontend), per video:

* the prompt: the same token ids and placeholder with either placement (asserted);
* the code frames on which the row's codes (device tokenizer encoder + host RVQ) agree with the front end's (all 20
  codes equal; asserted >= 95 %), the item's engine rows (frame rows bit-identical, audio rows' PCC);
* trace safety and DRAM: encoding the items compiles no program the warm-ups did not (asserted), leaves nothing
  allocated (asserted), and the per-chip DRAM peak of the audio part against the ViT's;
* time per placement: vLLM's front end (the loader's decode at 2 fps, then the served processor on a cache miss, of
  which the front-end tokenizer encoder) and the engine (tt/multimodal.encode_item on the row: ViT, tokenizer encoder,
  host RVQ, patch encoder), and the engine again for the same item with its embeddings gone from the EmbeddingCache
  (TRACK_CODES: no tokenizer run).

Two phases, so the front end's CPU work holds no device lock:

    # front end (no device): vLLM's loader with serving's --media-io-kwargs (2 fps, every frame) and the served
    # processor (generator_vllm.MiMoV2OmniTTProcessor) once per placement, each from a fresh decode; items to --out
    python models/experimental/mimo_v2_6_flash/tests/perf/video_audio_device.py front --out DIR VIDEO.mp4 ...
    # engine (one 1x8 row): the row's encoders as MiMoV2OmniForCausalLM.setup_multimodal builds them, warmed as
    # warmup_model_prefill warms them (all buckets, before any trace), then each saved item with each placement
    TT_VISIBLE_DEVICES=<the row's 8 chips> python .../video_audio_device.py engine --out DIR

Needs MIMO_MODEL_PATH with ``audio_tokenizer/``, vLLM 0.30, PyAV / OpenCV and torchaudio; MIMO_VISION_CACHE and
MIMO_AUDIO_CACHE for the converted weights.
"""

import argparse
import base64
import collections
import contextlib
import glob
import json
import os
import time

import torch
from loguru import logger

MIN_AGREEMENT = 0.95
PROMPT = "<|im_start|>user\n{video}Describe the video and what is said in it.<|im_end|>\n<|im_start|>assistant\n"
TIMES = collections.defaultdict(float)


@contextlib.contextmanager
def timed(name):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        TIMES[name] += time.perf_counter() - t0


def timing(owner, attr, name):
    fn = getattr(owner, attr)

    def wrapper(*args, **kwargs):
        with timed(name):
            return fn(*args, **kwargs)

    setattr(owner, attr, wrapper)


def front(a):
    os.environ.setdefault("VLLM_PLUGINS", "")
    from vllm.multimodal.media import MediaConnector

    from models.experimental.mimo_v2_6_flash.reference import vision_inputs as vi
    from models.experimental.mimo_v2_6_flash.reference.checkpoint import default_model_path
    from models.experimental.mimo_v2_6_flash.tt import video_audio
    from models.experimental.mimo_v2_6_flash.tt.audio import processor as audio_proc
    from models.experimental.mimo_v2_6_flash.tt.generator_vllm import (
        MiMoV2OmniTTDummyInputsBuilder,
        MiMoV2OmniTTProcessingInfo,
        MiMoV2OmniTTProcessor,
    )

    upstream = vi.vllm_processor(default_model_path())
    info = MiMoV2OmniTTProcessingInfo(upstream.info.ctx)
    proc = MiMoV2OmniTTProcessor(info, MiMoV2OmniTTDummyInputsBuilder(info))
    timing(audio_proc, "tokenize_mels", "tokenizer")
    conn = MediaConnector(media_io_kwargs={"video": {"num_frames": -1, "fps": 2}})
    os.makedirs(a.out, exist_ok=True)
    for path in a.videos:
        name = os.path.splitext(os.path.basename(path))[0]
        uri = "data:video/mp4;base64," + base64.b64encode(open(path, "rb").read()).decode()
        runs = {}
        for where in ("frontend", "device"):
            audio_proc.DEVICE_TOKENIZER = where == "device"
            video_audio._TRACKS.clear()  # each placement decodes the track again (a cache miss)
            TIMES.clear()
            with timed("decode"):
                video = conn.fetch_video(uri)
            with timed("processor"):
                out = proc(
                    PROMPT.format(video=vi.VIDEO_PLACEHOLDER),
                    proc.info.parse_mm_data({"video": [video]}),
                    None,
                    hf_processor_mm_kwargs={"mimo_video_audio": True},
                )
            (ph,) = out["mm_placeholders"]["video"]
            (item,) = out["mm_kwargs"]["video"]
            runs[where] = dict(
                ids=list(out["prompt_token_ids"]),
                ph=ph,
                fields={k: v.data for k, v in item.items()},
                times=dict(TIMES),
            )
            del out, item, video
            logger.info(f"{name} {where}: front end {json.dumps({k: round(v, 2) for k, v in TIMES.items()})}")
        f, d = runs["frontend"], runs["device"]
        assert d["ids"] == f["ids"], f"{name}: the prompt differs between placements"
        assert (d["ph"].offset, d["ph"].length) == (f["ph"].offset, f["ph"].length)
        assert "va_audio_codes" in f["fields"] and "va_audio_mel_bits" in d["fields"], (list(f), list(d))
        shared = set(f["fields"]) & set(d["fields"])
        assert all(torch.equal(f["fields"][k], d["fields"][k]) for k in shared), f"{name}: fields differ"
        torch.save(
            {
                "video": name,
                "prompt_tokens": len(f["ids"]),
                "offset": f["ph"].offset,
                "length": f["ph"].length,
                "is_embed": f["ph"].is_embed,
                "frontend": f["fields"],
                "device_only": {k: v for k, v in d["fields"].items() if k not in shared},
                "times": {"frontend": f["times"], "device": d["times"]},
            },
            os.path.join(a.out, f"{name}.pt"),
        )
        bits = d["fields"]["va_audio_mel_bits"]
        logger.info(
            f"{name}: {len(f['ids'])} prompt tokens either way; item {f['ph'].length} rows; audio {bits.shape[0] / 100:.0f} s"
            f" ({bits.numel() * 4 / 2**20:.0f} MB of mel bits, {f['fields']['va_audio_codes'].shape[0]} code frames)"
        )


def engine(a):
    import ttnn
    from models.experimental.mimo_v2_6_flash.reference.checkpoint import default_model_path
    from models.experimental.mimo_v2_6_flash.tests.common import pcc
    from models.experimental.mimo_v2_6_flash.tt import generator_vllm as gv
    from models.experimental.mimo_v2_6_flash.tt import multimodal as mm
    from models.experimental.mimo_v2_6_flash.tt.audio.host_backend import AudioCodesBackend
    from models.experimental.mimo_v2_6_flash.tt.common import fabric_config

    path = default_model_path()
    records = [torch.load(p, weights_only=False) for p in sorted(glob.glob(os.path.join(a.out, "*.pt")))]
    assert records, f"no items under {a.out} (run the front phase first)"
    ttnn.set_fabric_config(**fabric_config())
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=65536, trace_region_size=0)

    def dram_mb():
        mv = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        return mv.total_bytes_allocated_per_bank * mv.num_banks / 1e6

    peak = {"active": None}

    def tracked(fn):
        def wrapper(*args, **kwargs):
            out = fn(*args, **kwargs)
            if peak["active"]:
                peak[peak["active"]] = max(peak.get(peak["active"], 0.0), dram_mb())
            return out

        return wrapper

    try:
        t0 = time.time()
        # As MiMoV2OmniForCausalLM.setup_multimodal builds a replica's backend
        backend = AudioCodesBackend(
            gv.default_encoder_backend(path, mesh),
            path,
            device_patch=gv.audio_device_patch(path, mesh),
            device_tokenizer=gv.audio_device_tokenizer(path, mesh),
        )
        vit, tok, patch = backend.vision, backend.device_tokenizer, backend.device_patch
        assert tok is not None and patch is not None, "MIMO_AUDIO_TOKENIZER / MIMO_AUDIO_PATCH must be device"
        logger.info(f"encoders loaded in {time.time() - t0:.0f} s: {backend!r}; {dram_mb():.0f} MB per chip")
        t0 = time.time()
        vit.warm()
        tok.warm()
        patch.warm()
        programs, idle = mesh.num_program_cache_entries(), dram_mb()
        logger.info(f"warmed in {time.time() - t0:.0f} s: {programs} programs, {idle:.0f} MB per chip")
        timing(vit, "embed_video", "vit")
        timing(tok, "hidden", "tokenizer encoder")
        timing(tok.quantizer, "quantize", "host RVQ")
        timing(backend, "embed_audio_codes", "patch encoder")
        orig_vit, orig_codes = vit.embed_video, backend.audio_codes

        def vit_phase(*args, **kwargs):
            peak["active"] = "vit"
            try:
                return orig_vit(*args, **kwargs)
            finally:
                peak["active"] = None

        def audio_phase(*args, **kwargs):
            peak["active"] = "audio"
            try:
                return orig_codes(*args, **kwargs)
            finally:
                peak["active"] = None

        vit.embed_video, backend.audio_codes = vit_phase, audio_phase
        summary, items = [], []
        for r in records:
            name = r["video"]
            fields = {"frontend": r["frontend"]}
            fields["device"] = {k: v for k, v in r["frontend"].items() if k != "va_audio_codes"}
            fields["device"].update(r["device_only"])
            item_of = {}
            for key in ("frontend", "device"):
                entry = dict(
                    modality="video",
                    identifier=f"{name}-{key}",
                    offset=r["offset"],
                    length=r["length"],
                    is_embed=r["is_embed"],
                    data=fields[key],
                )
                (item_of[key],) = mm.parse_items([entry])
            items.append((name, item_of["device"]))
            rows, eng = {}, {}
            # "device again": the same item with its embeddings gone (the EmbeddingCache is the adapter's, not
            # encode_item's): its codes are remembered (TRACK_CODES), so no tokenizer run
            for where in ("frontend", "device", "device again"):
                TIMES.clear()
                with timed("total"):
                    rows[where] = mm.encode_item(backend, item_of[where.split()[0]])
                eng[where] = dict(TIMES)
                logger.info(f"{name} engine, {where}: {json.dumps({k: round(v, 2) for k, v in TIMES.items()})}")
            codes_f = r["frontend"]["va_audio_codes"].long()
            codes_d = mm.TRACK_CODES.get(f"{name}-device").long()
            assert codes_d.shape == codes_f.shape, (codes_d.shape, codes_f.shape)
            agree = (codes_d == codes_f).all(1).float().mean().item()
            is_embed = torch.as_tensor(r["is_embed"]).reshape(-1)
            n_audio = int((torch.as_tensor(r["frontend"]["video_audio_seg_lens"]).reshape(-1)).sum())
            audio_rows = _audio_rows(r)
            f_rows, d_rows = rows["frontend"], rows["device"]
            assert f_rows.shape == d_rows.shape == (int(is_embed.sum()), f_rows.shape[1]), (f_rows.shape, d_rows.shape)
            assert int(audio_rows.sum()) == n_audio
            frames_equal = torch.equal(f_rows[~audio_rows], d_rows[~audio_rows])
            audio_pcc = pcc(f_rows[audio_rows].float(), d_rows[audio_rows].float())
            assert torch.equal(rows["device again"], d_rows), f"{name}: remembered codes gave other rows"
            row = dict(
                video=name,
                prompt_tokens=r["prompt_tokens"],
                item_rows=r["length"],
                audio_tokens=n_audio,
                code_frames=codes_f.shape[0],
                agreement=round(100 * agree, 2),
                frame_rows_identical=frames_equal,
                audio_rows_pcc=round(audio_pcc, 5),
                front_s={w: round(t["decode"] + t["processor"], 1) for w, t in r["times"].items()},
                front_tokenizer_s=round(r["times"]["frontend"].get("tokenizer", 0.0), 1),
                engine_s={w: round(t["total"], 1) for w, t in eng.items()},
                engine_parts={w: {k: round(v, 2) for k, v in t.items() if k != "total"} for w, t in eng.items()},
            )
            summary.append(row)
            logger.info(f"SUMMARY {json.dumps(row)}")
            assert agree >= MIN_AGREEMENT, f"{name}: {100 * agree:.2f} % of the code frames agree"
        # DRAM peaks (sampled after every matmul / softmax / SDPA: slows the pass, so after the timed runs) and the
        # tokenizer's run-to-run determinism: each device item once more from its mel
        for owner, attr in ((ttnn, "matmul"), (ttnn, "softmax"), (ttnn.transformer, "scaled_dot_product_attention")):
            setattr(owner, attr, tracked(getattr(owner, attr)))
        first = {item.identifier: mm.TRACK_CODES.get(item.identifier) for _, item in items}
        for name, item in items:
            mm.TRACK_CODES.entries.clear()
            mm.TRACK_CODES.used = 0
            mm.encode_item(backend, item)
            again = mm.TRACK_CODES.get(item.identifier)
            assert torch.equal(again, first[item.identifier]), f"{name}: codes changed run to run"
        assert mesh.num_program_cache_entries() == programs, "an encode compiled programs the warm-ups did not"
        assert abs(dram_mb() - idle) < 0.01, f"an encode left {dram_mb() - idle:.1f} MB per chip allocated"
        logger.info(
            f"codes identical run to run; no program compiled after the warm-ups, nothing left allocated; per-chip "
            f"DRAM peak above idle (sampled): ViT {peak.get('vit', idle) - idle:.0f} MB, audio tokenizer encoder "
            f"{peak.get('audio', idle) - idle:.0f} MB"
        )
        for row in summary:
            print(json.dumps(row), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


def _audio_rows(r):
    """bool [num_embeds]: the item's embedding rows that are audio (vLLM's order: per group frames, then audio)."""
    grid = torch.as_tensor(r["frontend"]["video_grid_thw"]).reshape(-1)
    n_segs = int(torch.as_tensor(r["frontend"]["video_audio_n_segs"]).reshape(-1)[0])
    seg_lens = [int(x) for x in torch.as_tensor(r["frontend"]["video_audio_seg_lens"]).reshape(-1)[:n_segs]]
    per_group = int(grid[1] * grid[2]) // 4 * (int(grid[0]) // n_segs)
    flags = []
    for n in seg_lens:
        flags += [False] * per_group + [True] * n
    return torch.tensor(flags)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("front", "engine"))
    p.add_argument("videos", nargs="*")
    p.add_argument("--out", required=True)
    a = p.parse_intermixed_args()
    front(a) if a.phase == "front" else engine(a)


if __name__ == "__main__":
    main()
