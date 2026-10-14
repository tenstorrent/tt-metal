# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Pure tests for models.tt_dit.utils.vbench.vbench_prompt_list (no vbench / device needed).

vbench 0.1.5's custom_input mode accepts a one-element list for a single video file but, for a
directory, indexes ``prompt_list`` by filename (``prompt_list[path] for path in prompt_list``), so a
list there raises ``TypeError: list indices must be integers or slices, not str`` -- the failure the
LTX seed-averaged VBench gate hit in CI (T1 e2e job 104755168955).
"""

import os

from models.tt_dit.utils.vbench import vbench_prompt_list

PROMPT = "a girl singing"


def _vbench_directory_mode(videos_path, prompt_list):
    """Replay the exact vbench 0.1.5 directory-mode lines so the shape contract is tested, not assumed."""
    cur = [
        {"video_list": [os.path.join(videos_path, f)]}
        for f in os.listdir(videos_path)
        if os.path.splitext(f)[1].lower() in (".mp4", ".gif", ".jpg", ".png")
    ]
    prompt_list = {os.path.join(videos_path, path): prompt_list[path] for path in prompt_list}
    assert len(prompt_list) >= len(cur)
    video_map = {os.path.abspath(k): v for k, v in prompt_list.items()}
    return [video_map[os.path.abspath(v["video_list"][0])] for v in cur]


def test_single_file_is_one_element_list(tmp_path):
    clip = tmp_path / "ltx_av_fast_1920x1088_2.mp4"
    clip.write_bytes(b"")
    assert vbench_prompt_list(str(clip), PROMPT) == [PROMPT]


def test_missing_file_path_still_a_list(tmp_path):
    # The single-clip caller passes a filename that may not exist yet at shaping time; not a dir -> list.
    assert vbench_prompt_list(str(tmp_path / "nope.mp4"), PROMPT) == [PROMPT]


def test_none_prompt_is_empty_for_both_shapes(tmp_path):
    (tmp_path / "seed_0.mp4").write_bytes(b"")
    assert vbench_prompt_list(str(tmp_path), None) == []
    assert vbench_prompt_list(str(tmp_path / "seed_0.mp4"), None) == []


def test_directory_maps_every_video_to_the_prompt_and_skips_non_videos(tmp_path):
    for name in ("seed_0.mp4", "seed_1.mp4", "seed_2.MP4", "eval_full_info.json", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    got = vbench_prompt_list(str(tmp_path), PROMPT)
    assert got == {"seed_0.mp4": PROMPT, "seed_1.mp4": PROMPT, "seed_2.MP4": PROMPT}


def test_directory_shape_satisfies_vbench_directory_mode(tmp_path):
    for k in range(5):
        (tmp_path / f"seed_{k}.mp4").write_bytes(b"")
    got = vbench_prompt_list(str(tmp_path), PROMPT)
    assert _vbench_directory_mode(str(tmp_path), got) == [PROMPT] * 5


def test_list_shape_reproduces_the_ci_failure_in_directory_mode(tmp_path, expect_error):
    (tmp_path / "seed_0.mp4").write_bytes(b"")
    with expect_error(TypeError, "list indices must be integers"):
        _vbench_directory_mode(str(tmp_path), [PROMPT])
