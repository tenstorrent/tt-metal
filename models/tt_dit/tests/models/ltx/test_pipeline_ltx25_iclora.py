# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Owner-operated native fixture collection, not performance/quality qualification.

Run only through the device broker. Missing real assets fail explicitly. Retain
all media; compare to the manifest's baseline/criteria before claiming acceptance.
"""
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from PIL import Image

from .ltx_mesh_params import LTX_DISTILLED_MESH_PARAMS_DL


def _digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _now():
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z")


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp, dynamic_load",
    [p for p in LTX_DISTILLED_MESH_PARAMS_DL if p.id == "4x8sp1tp0nl2_ring_is_fsdp0"],
    indirect=["mesh_device", "device_params"],
)
def test_ltx25_iclora_output_dependent_collection(
    mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, dynamic_load
):
    from models.tt_dit.pipelines.ltx.pipeline_ltx25_iclora import LTX25ICLoRAPipeline
    from models.tt_dit.utils.video import export_video_audio

    fixture_path = os.environ.get("TT_ICLORA_FIXTURE")
    output_dir = os.environ.get("TT_ICLORA_OUTPUT")
    if not fixture_path or not output_dir:
        pytest.fail("native collection requires TT_ICLORA_FIXTURE and TT_ICLORA_OUTPUT; no real-weight skip")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=False)
    pipe = LTX25ICLoRAPipeline.create_pipeline(
        mesh_device,
        fixture_path=fixture_path,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        topology=topology,
        is_fsdp=is_fsdp,
        dynamic_load=dynamic_load,
        traced=True,
    )
    fixture = pipe.resolved_fixture
    if fixture["generation"]["mode"] != "r2v":
        pytest.fail("collection fixture must be r2v; the fourth call separately exercises both endpoints")
    (target / "runtime-options.json").write_text(
        json.dumps(
            dict(
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                num_links=num_links,
                topology=str(topology),
                is_fsdp=is_fsdp,
                dynamic_load=dynamic_load,
                traced=True,
            ),
            indent=2,
        )
        + "\n"
    )
    (target / "resolved-fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
    acceptance = fixture["acceptance"]
    pipe.warmup_buffers()
    first = acceptance["first_frame"]["path"]
    for index, seed in enumerate(acceptance["seeds"]):
        row = {
            "cell": f"r2v_{index}",
            "first_frame": first,
            "first_sha256": _digest(first),
            "seed": seed,
            "started": _now(),
            "quality_accepted": False,
        }
        try:
            frames, audio = pipe.generate_r2v(acceptance["prompt"], first_frame=first, seed=seed, output_type="float")
            row["generation_done"] = _now()
            g = fixture["generation"]
            assert tuple(frames.shape) == (1, 3, g["num_frames"], g["height"], g["width"])
            assert audio is not None
            output = target / f"r2v_{index}.mp4"
            export_video_audio(frames, str(output), fps=g["fps"], audio=audio)
            # Fresh decoded output, not an identical pre-cached input, drives the next call.
            first = str(target / f"r2v_{index}-last.png")
            pixels = ((frames[0, :, -1].float() + 1) * 127.5).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            Image.fromarray(pixels).save(first)
            row.update(
                output=str(output),
                output_sha256=_digest(output),
                next_first=first,
                next_first_sha256=_digest(first),
                file_ready=_now(),
                endpoint_cache_entries=len(pipe._pipeline._i2v_cond_cache),
            )
        except BaseException as error:
            row.update(error=type(error).__name__ + ": " + str(error), failed=_now())
            raise
        finally:
            with (target / "calls.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
    output = target / "fl2v.mp4"
    row = {"cell": "fl2v", "started": _now(), "quality_accepted": False}
    try:
        result = pipe.generate_r2v(
            acceptance["prompt"],
            first_frame=acceptance["first_frame"]["path"],
            last_frame=acceptance["last_frame"]["path"],
            seed=acceptance["seeds"][0],
            output_path=str(output),
        )
        assert result == str(output)
        row.update(output=result, output_sha256=_digest(output), file_ready=_now())
    except BaseException as error:
        row.update(error=type(error).__name__ + ": " + str(error), failed=_now())
        raise
    finally:
        with (target / "calls.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
