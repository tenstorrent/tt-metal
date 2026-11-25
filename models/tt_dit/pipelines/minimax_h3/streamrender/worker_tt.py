# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""StreamRender-H3 worker with the DiT on a Blackhole galaxy.

The single-process counterpart of Sana's `runtime/worker.py`: the same loopback HTTP bridge and session loop,
with the 50-layer DiT on a 4x8 mesh (TP=4 on axis 0, SP=8 on axis 1) and everything else on the host.

    python -m models.tt_dit.pipelines.minimax_h3.streamrender.worker_tt \
        --sana-root .../streamrender_h3 --snapshot .../snapshot --assets .../assets.tt.json --smoke 4

`--encode-qwen` runs one session start with the text encoder (no device) to cache the Qwen context. Runs need
their own `TT_DIT_CACHE_DIR`: the converted weights share the base checkpoint's subfolder name.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import queue
import threading
import time
from pathlib import Path

import torch

import ttnn

from ....models.transformers.minimax_h3.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils import cache
from ....utils.test import create_fabric_router_config
from .dit_adapter import PrefixBank, PrefixCapture, StreamRenderDiT
from .sana_host import build_host_models, make_meta_class, make_pipeline_class, mount_sana, qwen_cache_dir

logger = logging.getLogger("streamrender.tt")

MODEL_NAME = "minimax-h3-streamrender"


def open_mesh(shape: tuple[int, int]) -> ttnn.MeshDevice:
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        create_fabric_router_config(),
    )
    # The trace region holds the two continuation traces (one per prefix bank).
    return ttnn.open_mesh_device(ttnn.MeshShape(*shape), l1_small_size=65536, trace_region_size=1_200_000_000)


def build_dit(
    mesh_device: ttnn.MeshDevice, snapshot: Path, *, trace: bool = False, tp_axis: int = 0
) -> StreamRenderDiT:
    """The DiT with TP on `tp_axis` of the mesh and SP on the other."""
    from safetensors.torch import load_file

    subfolder = snapshot / "transformer_ref"
    config = json.loads((subfolder / "config.json").read_text())
    rope_freq_dim, rope_theta = config["rope_freq_dim"], config["rope_theta"]
    config = {k: v for k, v in config.items() if not k.startswith("_") and k not in ("rope_freq_dim", "rope_theta")}
    config["patch_size"] = tuple(config["patch_size"])
    sp_axis = 1 - tp_axis
    tp, sp = tuple(mesh_device.shape)[tp_axis], tuple(mesh_device.shape)[sp_axis]
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp),
        cfg_parallel=None,
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=2, topology=ttnn.Topology.Ring)
    transformer = MiniMaxH3Transformer3DModel(
        **config, mesh_device=mesh_device, ccl_manager=ccl_manager, parallel_config=parallel_config
    )

    def state_dict() -> dict[str, torch.Tensor]:
        index = json.loads((subfolder / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
        state: dict[str, torch.Tensor] = {}
        for shard in sorted(set(index.values())):
            state.update(load_file(str(subfolder / shard)))
        return state

    started = time.perf_counter()
    cache.load_model(
        transformer,
        model_name=MODEL_NAME,
        subfolder="transformer_ref_resident_adaln",
        parallel_config=parallel_config,
        mesh_shape=tuple(mesh_device.shape),
        mesh_device=mesh_device,
        get_torch_state_dict=state_dict,
    )
    logger.info("DiT weights on device in %.0f s", time.perf_counter() - started)
    return StreamRenderDiT(
        transformer,
        mesh_device=mesh_device,
        sp_axis=sp_axis,
        sp_factor=sp,
        rope_freq_dim=rope_freq_dim,
        rope_theta=rope_theta,
        head_dim=config["attention_head_dim"],
        trace=trace,
    )


class ZeroDiT:
    """Host-only stand-in for `StreamRenderDiT`: zero velocity, and it logs every forward's packed row counts."""

    def __call__(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        cache = kwargs.get("past_key_values")
        rows = {k: kwargs[k]["position_ids"].numel() for k in ("text_pos_info", "img_pos_info", "audio_pos_info")}
        mode = "prefix" if isinstance(cache, PrefixCapture) else "suffix" if isinstance(cache, PrefixBank) else "full"
        logger.info("dry-run %s forward: %d rows %s", mode, kwargs["x"].shape[1], rows)
        if mode == "prefix":
            cache.bank = PrefixBank(slots=[], rows=kwargs["x"].shape[1], padded_rows=kwargs["x"].shape[1])
        video = kwargs["img_pos_for_infer_output_info"]["position_ids"].numel()
        return kwargs["x"].new_zeros(video, kwargs["x"].shape[-1]), kwargs["audio_x"].new_zeros(
            rows["audio_pos_info"], kwargs["audio_x"].shape[-1]
        )


def _frames_uint8(rgb: torch.Tensor):
    """`(1, 3, T, H, W)` decoder output -- uint8 RGB, or [0, 1] floats -- as T `(H, W, 3)` uint8 frames."""
    frames = rgb[0].permute(1, 2, 3, 0)
    return (frames if frames.dtype == torch.uint8 else frames.clamp(0, 1).mul(255).byte()).numpy()


class EncodeAhead:
    """Plans, gathers and TAE-encodes chunks on a thread, one chunk ahead of the device.

    `gather(frames)` returns `(rgb, extra)` for the next chunk or `None` to end the stream. The encoder is CPU torch,
    which releases the GIL in its kernels, so it overlaps the previous chunk's DiT and decoder.
    """

    def __init__(self, pipe, gather) -> None:
        self._pipe, self._gather = pipe, gather
        self._chunks: queue.Queue = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._run, name="encode-ahead", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while True:
                count, frames = self._pipe.plan_next()
                gathered = self._gather(frames)
                if gathered is None:
                    self._chunks.put(None)
                    return
                rgb, extra = gathered
                started = time.perf_counter()
                encoded = self._pipe.encode(rgb, count)
                self._chunks.put((rgb, encoded, extra, (time.perf_counter() - started) * 1000))
        except BaseException as error:
            self._chunks.put(error)

    def next(self):
        """`(rgb, encoded, extra, encode_ms)` of the next chunk, or `None` at the end of the stream."""
        item = self._chunks.get()
        if isinstance(item, BaseException):
            raise item
        return item


def run_smoke(pipe, settings, assets, rounds: int, dump: Path | None, seed: int) -> None:
    from PIL import Image

    pipe.start(prompt=assets["prompt"], picture=assets["reference_image"], seed=seed)
    reference = settings["reference"]
    remaining = [rounds]

    def zeros(frames):
        if remaining[0] == 0:
            return None
        remaining[0] -= 1
        return torch.zeros((3, frames, reference["height"], reference["width"])), None

    ahead = EncodeAhead(pipe, zeros)
    while (chunk := ahead.next()) is not None:
        rgb_in, encoded, _, encode_ms = chunk
        rgb, metrics = pipe.render_encoded(rgb_in, encoded)
        metrics["encode_ms"] = encode_ms
        print("STREAMRENDER_CHUNK", json.dumps(metrics), flush=True)
        if dump is not None and rgb is not None:
            dump.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(_frames_uint8(rgb)):
                Image.fromarray(frame).save(dump / f"r{metrics['round']:03d}_{i:02d}.jpg", quality=90)


def serve_sessions(pipe, settings, assets, seed: int) -> None:
    """Sana's `run_h3` session loop for one process.

    JPEG encoding and hand-off run on a publisher thread, as in Sana's worker, so the device starts the next chunk
    while the previous one is still being encoded. One chunk is in flight at a time, which keeps output order.
    """
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    from PIL import Image
    from runtime.server import Bridge, serve

    def publish(session, rgb, metrics):
        images = []
        for frame in _frames_uint8(rgb):
            out = io.BytesIO()
            Image.fromarray(frame).save(out, format="JPEG", quality=85)
            images.append(out.getvalue())
        while not session.closed:
            try:
                session.outputs.put((images, metrics), timeout=0.5)
                break
            except queue.Full:
                continue

    bridge = Bridge(settings)
    httpd = serve(bridge)
    publisher = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rgb-publication")
    try:
        bridge.worker_ready = True
        print("STREAMRENDER_READY", json.dumps({"backend": "tt", "layers": 50, "nfe": 2}), flush=True)
        while True:
            session = bridge.starts.get()
            if session is None:
                return
            pipe.start(prompt=assets["prompt"], picture=assets["reference_image"], seed=seed)
            session.ready = True
            pending = None

            def gather(frames, session=session):
                if session.closed:
                    return None
                records = session.collect(frames, settings["session_idle_seconds"])
                if records is None:
                    return None
                arrays = [np.array(Image.open(io.BytesIO(png)).convert("RGB")) for _, png, _, _ in records]
                pixels = torch.from_numpy(np.stack(arrays)).permute(3, 0, 1, 2).float().div_(255)
                # Semantic palette labels must not be bilinearly blended.
                pixels = (
                    torch.nn.functional.interpolate(
                        pixels.permute(1, 0, 2, 3),
                        size=(settings["reference"]["height"], settings["reference"]["width"]),
                        mode="nearest",
                    )
                    .permute(1, 0, 2, 3)
                    .contiguous()
                )
                return pixels, records

            ahead = EncodeAhead(pipe, gather)
            while (chunk := ahead.next()) is not None:
                pixels, encoded, records, encode_ms = chunk
                rgb, metrics = pipe.render_encoded(pixels, encoded)
                metrics["encode_ms"] = encode_ms
                metrics["session"] = session.id
                metrics["oldest_input_to_rgb_ready_ms"] = (time.monotonic() - records[0][3]) * 1000
                session.metrics = metrics
                print("STREAMRENDER_CHUNK", json.dumps(metrics), flush=True)
                if rgb is None:
                    continue
                if pending is not None:
                    pending.result()
                pending = publisher.submit(publish, session, rgb, metrics)
            if pending is not None:
                pending.result()
    except BaseException as error:
        bridge.fail(error)
        raise
    finally:
        publisher.shutdown(wait=True)
        httpd.shutdown()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sana-root", type=Path, required=True, help="Sana models/streamrender_h3 checkout")
    parser.add_argument("--snapshot", type=Path, help="converted MiniMax-H3 snapshot (convert_backbone.py)")
    parser.add_argument("--assets", type=Path, required=True, help="Sana deployment asset manifest")
    parser.add_argument("--config", type=Path, help="Sana runtime config (default: <sana-root>/configs/runtime.json)")
    parser.add_argument("--port", type=int)
    parser.add_argument("--mesh", default="4x8")
    parser.add_argument("--encode-qwen", action="store_true", help="cache the Qwen context and exit (host only)")
    parser.add_argument("--smoke", type=int, default=0, help="render this many zero-input chunks and exit")
    parser.add_argument("--dump", type=Path, help="write smoke frames here")
    parser.add_argument("--dry-run", action="store_true", help="zero-output DiT on the host; no device")
    parser.add_argument("--trace-dit", action="store_true", help="replay continuations from captured traces")
    parser.add_argument("--dit-tp-axis", type=int, default=0, help="mesh axis of the DiT's tensor parallelism")
    args = parser.parse_args()

    mount_sana(args.sana_root)
    from dev.yanzuolu.common.config import CfgNode
    from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
    from runtime.config import load_config

    settings, assets = load_config(args.config or args.sana_root / "configs/runtime.json", args.assets)
    # `load_config` resolves only `asset://` references in the config; manifest paths are relative to the manifest.
    assets["reference_image"] = str((args.assets.resolve().parent / assets["reference_image"]).resolve())
    if args.port:
        settings["port"] = args.port
    cfg = CfgNode(settings["inference"])
    meta_class = make_meta_class()
    meta_class.qwen_cache_dir = qwen_cache_dir()
    pipeline_class = make_pipeline_class()

    mesh_device = None
    try:
        if args.encode_qwen:
            models = build_host_models(cfg, None, text_encoder=True)
        elif args.dry_run:
            models = build_host_models(cfg, ZeroDiT(), text_encoder=False)
        else:
            if args.snapshot is None:
                parser.error("--snapshot is required unless --encode-qwen")
            mesh_device = open_mesh(tuple(int(v) for v in args.mesh.split("x")))
            models = build_host_models(
                cfg,
                build_dit(mesh_device, args.snapshot, trace=args.trace_dit, tp_axis=args.dit_tp_axis),
                text_encoder=False,
                mesh_device=mesh_device,
            )
            # Load the decoder before the first session rather than inside its first chunk.
            output = settings["output"]
            models["video_decoder"].fast_for(output["height"] // 16, output["width"] // 16, torch.device("cpu"))
        pipe = pipeline_class(cfg, models, settings, meta_class)
        with execution_phase(ExecutionPhase.VALIDATION):
            if args.encode_qwen:
                pipe.start(prompt=assets["prompt"], picture=assets["reference_image"], seed=cfg.seed)
                logger.info("Qwen context cached under %s", meta_class.qwen_cache_dir)
            elif args.smoke:
                run_smoke(pipe, settings, assets, args.smoke, args.dump, cfg.seed)
            else:
                serve_sessions(pipe, settings, assets, cfg.seed)
    finally:
        if mesh_device is not None:
            ttnn.close_mesh_device(mesh_device)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
