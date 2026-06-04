# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch

from models.experimental.audiox.demo import stage3_switch as switch_mod


def test_build_cases_defaults_to_synthetic_visual(tmp_path):
    args = switch_mod._parse_args(
        [
            "--checkpoint",
            "/tmp/fake.safetensors",
            "--output-dir",
            str(tmp_path),
        ]
    )

    cases = switch_mod._build_cases(args)

    assert [case["name"] for case in cases] == [
        "text_to_audio",
        "text_to_music",
        "video_to_audio",
        "video_to_music",
    ]
    assert cases[2]["video_path"] is None
    assert cases[2]["video_prompt_tensor"] is True
    assert cases[3]["video_prompt_tensor"] is True


def test_main_reuses_single_session_and_writes_summary(tmp_path, monkeypatch):
    session_inits = []
    session_runs = []
    device_tokens = []

    class FakeSession:
        def __init__(self, checkpoint, device, seed=None):
            session_inits.append((str(checkpoint), device, seed))

        def run(
            self,
            prompt,
            output,
            *,
            video_path=None,
            video_prompt_tensor=None,
            steps=0,
            seed=0,
            duration_seconds=None,
            return_details=False,
        ):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"fake")
            session_runs.append(
                {
                    "prompt": prompt,
                    "output": str(output),
                    "video_path": None if video_path is None else str(video_path),
                    "synthetic": video_prompt_tensor is not None,
                    "steps": steps,
                    "duration_seconds": duration_seconds,
                }
            )
            return {
                "output_path": output,
                "latent": torch.zeros(1, 4, 4),
                "cross_attn_cond": torch.zeros(1, 3, 8),
                "conditioning_tokens": 3,
                "t_latent": 4,
                "duration_seconds": duration_seconds,
                "timings": {
                    "sampling_seconds": 0.5,
                    "generation_seconds": 2.0,
                },
            }

        def deallocate(self):
            pass

    def fake_open_tt_device(device_id):
        device_tokens.append(("open", device_id))
        return f"device-{device_id}"

    def fake_close_tt_device(device):
        device_tokens.append(("close", device))

    def fake_build_synthetic_video_prompt(args):
        return torch.zeros(1, 5, 3, 224, 224)

    def fake_summarize_run_details(output_path, elapsed_seconds, details):
        return {
            "path": str(output_path),
            "valid_16khz": True,
            "elapsed_seconds": elapsed_seconds,
            "generation_seconds": details["timings"]["generation_seconds"],
            "conditioning_tokens": details["conditioning_tokens"],
            "latent_tokens": details["t_latent"],
            "diffusion_token_steps": details["t_latent"] * details["steps"],
            "sampling_seconds": details["timings"]["sampling_seconds"],
            "diffusion_tokens_per_second": 8.0,
        }

    monkeypatch.setattr(switch_mod, "_build_session", lambda checkpoint, device, seed: FakeSession(checkpoint, device, seed))
    monkeypatch.setattr(switch_mod, "_open_tt_device", fake_open_tt_device)
    monkeypatch.setattr(switch_mod, "_close_tt_device", fake_close_tt_device)
    monkeypatch.setattr(switch_mod, "_build_synthetic_video_prompt", fake_build_synthetic_video_prompt)
    monkeypatch.setattr(switch_mod.validate_mod, "_summarize_run_details", fake_summarize_run_details)

    rc = switch_mod.main(
        [
            "--checkpoint",
            "/tmp/fake.safetensors",
            "--output-dir",
            str(tmp_path),
            "--steps",
            "2",
            "--warmup-runs",
            "1",
        ]
    )

    assert rc == 0
    assert len(session_inits) == 1
    assert device_tokens == [("open", 0), ("close", "device-0")]
    assert len(session_runs) == 5
    summary = (tmp_path / "stage3_switch_summary.json").read_text()
    assert "text_to_audio" in summary
    assert "video_to_music" in summary
