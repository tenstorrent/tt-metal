# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for coordinate overlay helpers — written BEFORE the implementation (TDD)."""


from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _white_image(w=200, h=200):
    return Image.new("RGB", (w, h), color=(255, 255, 255))


# ---------------------------------------------------------------------------
# annotate_image_with_points
# ---------------------------------------------------------------------------


def test_annotate_image_with_points_creates_output_file(tmp_path):
    """Calling annotate_image_with_points saves an annotated PNG to output_path."""
    from models.demos.molmo2.demo.demo import annotate_image_with_points

    img = _white_image(200, 200)
    out = tmp_path / "out.png"
    # coords "500 500" → (100px, 100px) on a 200×200 image
    annotate_image_with_points(img, "0 500 500", out)
    assert out.exists(), "Output file must be created"


def test_annotate_image_with_points_draws_visible_mark(tmp_path):
    """The annotated image has a coloured dot at the coordinate position."""
    from models.demos.molmo2.demo.demo import annotate_image_with_points

    img = _white_image(200, 200)
    out = tmp_path / "out.png"
    # coords "500 500" → centre of 200×200 image
    annotate_image_with_points(img, "0 500 500", out)
    result = Image.open(out).convert("RGB")
    # Centre pixel should no longer be pure white — a dot was drawn there
    cx, cy = 100, 100
    r, g, b = result.getpixel((cx, cy))
    assert (r, g, b) != (255, 255, 255), "A coloured mark must appear at the coordinate"


def test_annotate_image_with_points_molmo_format(tmp_path):
    """Handles actual Molmo coord format 'img_idx x_3dig y_3dig' (x/y are 3–4 digit padded)."""
    from models.demos.molmo2.demo.demo import annotate_image_with_points

    img = _white_image(1000, 1000)
    out = tmp_path / "out.png"
    # Actual model output format: "1 070 576" → x=70/1000=7%, y=576/1000=57.6%
    # with a spurious leading "1 " (like "1 1 070 576") the parser must skip the leading 1
    annotate_image_with_points(img, "1 1 070 576", out)
    result = Image.open(out).convert("RGB")
    # Expected dot at (70, 576) on 1000×1000 image
    assert result.getpixel((70, 576)) != (
        255,
        255,
        255,
    ), "Dot must appear at x=70, y=576 (Molmo 3-digit format: idx=1 x=070 y=576)"


def test_annotate_image_with_points_multiple_coords(tmp_path):
    """Multiple space-separated coord triplets each produce a visible mark."""
    from models.demos.molmo2.demo.demo import annotate_image_with_points

    img = _white_image(400, 400)
    out = tmp_path / "out.png"
    # Two points: (0, 250, 250) → (100, 100) and (0, 750, 750) → (300, 300)
    annotate_image_with_points(img, "0 250 250 0 750 750", out)
    result = Image.open(out).convert("RGB")
    assert result.getpixel((100, 100)) != (255, 255, 255), "Mark at first point"
    assert result.getpixel((300, 300)) != (255, 255, 255), "Mark at second point"


def test_annotate_image_with_points_preserves_image_size(tmp_path):
    """The output image has the same dimensions as the input."""
    from models.demos.molmo2.demo.demo import annotate_image_with_points

    img = _white_image(320, 240)
    out = tmp_path / "out.png"
    annotate_image_with_points(img, "0 500 500", out)
    result = Image.open(out)
    assert result.size == (320, 240), "Image dimensions must be unchanged"


# ---------------------------------------------------------------------------
# annotate_video_frame_with_points
# ---------------------------------------------------------------------------


def test_annotate_video_frame_with_points_creates_output_file(tmp_path):
    """annotate_video_frame_with_points saves a single annotated frame PNG."""
    from models.demos.molmo2.demo.demo import annotate_video_frame_with_points

    frame = _white_image(320, 240)
    out = tmp_path / "frame.png"
    annotate_video_frame_with_points(frame, x_norm=500, y_norm=500, output_path=out)
    assert out.exists(), "Frame output file must be created"


def test_annotate_video_frame_with_points_draws_mark(tmp_path):
    """The saved frame has a coloured dot at the normalised coordinate."""
    from models.demos.molmo2.demo.demo import annotate_video_frame_with_points

    frame = _white_image(200, 200)
    out = tmp_path / "frame.png"
    annotate_video_frame_with_points(frame, x_norm=500, y_norm=500, output_path=out)
    result = Image.open(out).convert("RGB")
    assert result.getpixel((100, 100)) != (255, 255, 255), "Mark at centre"


# ---------------------------------------------------------------------------
# save_annotated_outputs  (integration — no hardware, pure PIL)
# ---------------------------------------------------------------------------


def test_save_annotated_outputs_image_creates_file(tmp_path):
    """save_annotated_outputs writes an annotated image when given image input."""
    from models.demos.molmo2.demo.demo import save_annotated_outputs

    img = _white_image(200, 200)
    img_path = tmp_path / "input.png"
    img.save(img_path)

    coords_text = '<points coords="0 500 500"/>'
    out_dir = tmp_path / "annotated"
    save_annotated_outputs(
        prompts=[
            [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": str(img_path)}, {"type": "text", "text": "Point"}],
                }
            ]
        ],
        responses=[coords_text],
        output_dir=out_dir,
    )
    files = list(out_dir.glob("*.png"))
    assert len(files) >= 1, "At least one annotated image must be saved"


def test_save_annotated_outputs_skips_text_only(tmp_path):
    """save_annotated_outputs does nothing for text-only responses."""
    from models.demos.molmo2.demo.demo import save_annotated_outputs

    out_dir = tmp_path / "annotated"
    save_annotated_outputs(
        prompts=[[{"role": "user", "content": "What is 2+2?"}]],
        responses=["4"],
        output_dir=out_dir,
    )
    # No output directory should be created for text-only
    assert not out_dir.exists() or len(list(out_dir.glob("*"))) == 0


# ---------------------------------------------------------------------------
# annotate_video_with_points  (new: produces annotated .mp4)
# ---------------------------------------------------------------------------

import numpy as np


def _make_test_video(path, n_frames=10, w=64, h=64, fps=2.0):
    """Write a minimal solid-colour MP4 for testing (uses PyAV directly)."""
    import av

    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=int(round(fps)))
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    for _ in range(n_frames):
        frame_arr = np.full((h, w, 3), 200, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


def test_annotate_video_with_points_creates_mp4(tmp_path):
    """annotate_video_with_points writes an .mp4 file."""
    from models.demos.molmo2.demo.demo import annotate_video_with_points

    vid = tmp_path / "in.mp4"
    _make_test_video(vid, n_frames=10, fps=2.0)
    out = tmp_path / "out.mp4"
    # coords: t=0.0, x=500, y=500 → centre of 64×64 frame
    annotate_video_with_points(str(vid), "0.0 1 500 500", str(out))
    assert out.exists(), "Output MP4 must be created"


def _read_video_frames(path):
    """Read all frames from an MP4 using PyAV, return list of numpy arrays."""
    import av

    frames = []
    container = av.open(str(path))
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return frames


def test_annotate_video_with_points_output_has_same_frame_count(tmp_path):
    """Annotated video has the same number of frames as the input."""
    from models.demos.molmo2.demo.demo import annotate_video_with_points

    vid = tmp_path / "in.mp4"
    _make_test_video(vid, n_frames=8, fps=2.0)
    out = tmp_path / "out.mp4"
    annotate_video_with_points(str(vid), "0.0 1 500 500", str(out))
    frames = _read_video_frames(out)
    assert len(frames) == 8, f"Expected 8 frames, got {len(frames)}"


def test_annotate_video_with_points_draws_dot_on_matching_frame(tmp_path):
    """A dot is drawn on the frame closest to the given timestamp."""
    from models.demos.molmo2.demo.demo import annotate_video_with_points

    vid = tmp_path / "in.mp4"
    _make_test_video(vid, n_frames=6, fps=2.0)  # frames at t=0,0.5,1.0,1.5,2.0,2.5
    out = tmp_path / "out.mp4"
    annotate_video_with_points(str(vid), "0.0 1 500 500", str(out))
    frames = _read_video_frames(out)
    cx, cy = 32, 32  # centre of 64×64
    r, g, b = frames[0][cy, cx]
    assert (int(r), int(g), int(b)) != (200, 200, 200), "Dot must be drawn on frame 0"


def test_annotate_video_dot_uses_nearest_timestamp(tmp_path):
    """With two tracking timestamps, each half of the video uses the nearest one."""
    from models.demos.molmo2.demo.demo import annotate_video_with_points

    # 20-frame video at 10fps → 2s. Two coords: t=0.0 top-left, t=1.5 bottom-right.
    vid = tmp_path / "in.mp4"
    _make_test_video(vid, n_frames=20, w=64, h=64, fps=10.0)
    out = tmp_path / "out.mp4"
    # t=0.0 → dot at top-left (100,100); t=1.5 → dot at bottom-right (900,900)
    annotate_video_with_points(str(vid), "0.0 1 100 100;1.5 1 900 900", str(out))
    frames = _read_video_frames(out)
    # Frame 0 (t=0.0) → nearest is t=0.0 → dot at top-left (6,6)
    assert tuple(int(v) for v in frames[0][6, 6]) != (200, 200, 200), "Frame 0: dot at top-left"
    # Frame 19 (t=1.9) → nearest is t=1.5 → dot at bottom-right (57,57)
    assert tuple(int(v) for v in frames[19][57, 57]) != (200, 200, 200), "Frame 19: dot at bottom-right"


def test_save_annotated_outputs_video_produces_mp4(tmp_path):
    """save_annotated_outputs writes an annotated .mp4 for video prompts."""
    from models.demos.molmo2.demo.demo import save_annotated_outputs

    vid = tmp_path / "test.mp4"
    _make_test_video(vid, n_frames=6, fps=2.0)

    out_dir = tmp_path / "out"
    save_annotated_outputs(
        prompts=[
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": str(vid)},
                        {"type": "text", "text": "Track"},
                    ],
                }
            ]
        ],
        responses=['<tracks coords="0.0 1 500 500;0.5 1 500 500"/>'],
        output_dir=out_dir,
    )
    mp4s = list(out_dir.glob("*.mp4"))
    assert len(mp4s) >= 1, "At least one annotated .mp4 must be saved"


# ---------------------------------------------------------------------------
# Truncated output handling (max_new_tokens hit before closing quote)
# ---------------------------------------------------------------------------


def test_save_annotated_outputs_handles_truncated_track_coords(tmp_path):
    """save_annotated_outputs extracts coords even when the closing quote is missing."""
    from models.demos.molmo2.demo.demo import save_annotated_outputs

    vid = tmp_path / "test.mp4"
    _make_test_video(vid, n_frames=6, fps=2.0)
    out_dir = tmp_path / "out"
    # Simulate truncated model output — no closing " or />
    truncated = '<tracks coords="0.0 1 500 500;0.5 1 500 500'
    save_annotated_outputs(
        prompts=[
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": str(vid)},
                        {"type": "text", "text": "Track"},
                    ],
                }
            ]
        ],
        responses=[truncated],
        output_dir=out_dir,
    )
    mp4s = list(out_dir.glob("*.mp4"))
    assert len(mp4s) >= 1, "Annotated MP4 must be saved even for truncated track output"


# ---------------------------------------------------------------------------
# Dot size
# ---------------------------------------------------------------------------


def test_dot_radius_is_at_least_150px(tmp_path):
    """The drawn dot covers pixels at least 150px from the centre (10x bigger for visibility)."""
    from models.demos.molmo2.demo.demo import annotate_image_with_points

    img = _white_image(500, 500)
    out = tmp_path / "out.png"
    annotate_image_with_points(img, "0 500 500", out)  # dot at (250, 250)
    result = Image.open(out).convert("RGB")
    assert result.getpixel((250 + 150, 250)) != (255, 255, 255), "Dot must extend ≥150px from centre"


# ---------------------------------------------------------------------------
# Video coord parser: multi-object same frame, persistent dot
# ---------------------------------------------------------------------------


def test_annotate_video_multiobject_draws_both_dots(tmp_path):
    """Two objects at the same timestamp both get dots (format: t obj_idx x y obj_idx x y)."""
    from models.demos.molmo2.demo.demo import annotate_video_with_points

    # 64×64 white video, 10 frames at 1fps
    vid = tmp_path / "in.mp4"
    _make_test_video(vid, n_frames=10, w=64, h=64, fps=1.0)
    out = tmp_path / "out.mp4"
    # Frame 2 (t=2.0): two objects — top-left and bottom-right
    annotate_video_with_points(str(vid), "2.0 1 100 100 2 900 900", str(out))
    frames = _read_video_frames(out)
    # Top-left quadrant (10,10) should have dot
    r, g, b = frames[2][10, 10]
    assert (int(r), int(g), int(b)) != (200, 200, 200), "First dot must appear (top-left)"
    # Bottom-right quadrant (54,54) should have dot
    r, g, b = frames[2][54, 54]
    assert (int(r), int(g), int(b)) != (200, 200, 200), "Second dot must appear (bottom-right)"


def test_annotate_video_dot_persists_across_frames(tmp_path):
    """The dot is visible on every frame, not just the nearest-timestamp frame."""
    from models.demos.molmo2.demo.demo import annotate_video_with_points

    vid = tmp_path / "in.mp4"
    _make_test_video(vid, n_frames=20, w=64, h=64, fps=10.0)  # 2s video at 10fps
    out = tmp_path / "out.mp4"
    # Single coord at t=1.0 — dot should stay visible on all frames
    annotate_video_with_points(str(vid), "1.0 1 500 500", str(out))
    frames = _read_video_frames(out)
    cx, cy = 32, 32
    dotted_frames = sum(1 for f in frames if tuple(int(v) for v in f[cy, cx]) != (200, 200, 200))
    # All 20 frames should show the dot (nearest-coord policy)
    assert dotted_frames == 20, f"Dot must persist on all frames, got {dotted_frames}/20"


def test_pointing_dot_only_visible_near_annotated_timestamp(tmp_path):
    """For single-timestamp pointing, dot only appears near that timestamp, not whole video."""
    from models.demos.molmo2.demo.demo import annotate_video_with_points

    # 40-frame video at 10fps = 4s. Single point at t=3.0
    vid = tmp_path / "in.mp4"
    _make_test_video(vid, n_frames=40, w=64, h=64, fps=10.0)
    out = tmp_path / "out.mp4"
    annotate_video_with_points(str(vid), "3.0 1 500 500", str(out))
    frames = _read_video_frames(out)
    cx, cy = 32, 32
    # Frame 0 (t=0.0) is far from t=3.0 — should NOT have dot
    assert tuple(int(v) for v in frames[0][cy, cx]) == (
        200,
        200,
        200,
    ), "Frame at t=0 must not show dot (too far from annotated t=3.0)"
    # Frame 30 (t=3.0) — should have dot
    assert tuple(int(v) for v in frames[30][cy, cx]) != (200, 200, 200), "Frame at t=3.0 must show dot"
