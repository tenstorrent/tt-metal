# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import itertools
import os
from pathlib import Path

import numpy as np
import pytest
from loguru import logger
from PIL import Image, ImageFilter

import ttnn
from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline

from .device_params import EDIT_MESH_DEVICE_PARAMS, MESH_DEVICE_PARAMS

DOG_IMAGE_PATH = Path(__file__).resolve().parents[4] / "demos" / "multimodal" / "gemma3" / "dog.jpg"


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize(
    "traced",
    [
        pytest.param(True, id="traced"),
        # pytest.param(False, id="not_traced"),
    ],
)
def test_qwenimage_pipeline(
    *,
    mesh_device: ttnn.MeshDevice,
    no_prompt: bool,
    traced: bool,
    is_ci_env: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Set TT_DIT_CACHE in CI environment. The path is needed for dynamic load when necessary.
    if is_ci_env:
        monkeypatch.setenv("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")

    width, height = 1024, 1024

    pipeline = QwenImagePipeline.create_pipeline(mesh_device=mesh_device)

    prompts = [
        'A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light '
        'beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the '
        'poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition'
        ", Ultra HD, 4K, cinematic composition.",
        'Tokyo neon alley at night, rain-slick pavement, cinematic cyberpunk lighting; include glowing sign text "深夜営業" in bold neon above a doorway; moody reflections, shallow depth of field.',
        'Steamy ramen shop entrance at dusk; fabric noren curtain gently swaying; print "しょうゆラーメン" across the curtain in thick brush-style kana; warm lantern light, photorealistic.',
        'Minimalist tea poster, cream background, elegant layout; vertical calligraphy "抹茶" centered in sumi ink; small red hanko-style seal "本格" in the corner; high-resolution graphic design.',
        'Hardcover fantasy novel cover, textured paper, gold foil; title text "物語のはじまり" centered; author line "山本ひかり" below; tasteful serif typography, dramatic vignette illustration.',
        'Anime manga panel, dynamic action lines; speech bubble with clear Japanese text "大丈夫、行こう！"; bold hand-lettered style; black-and-white screentone shading.',  # noqa: RUF001
        'Shinto shrine ema (wooden wish plaque) close-up; handwritten ink message "合格祈願"; tied with a red cord; shallow depth of field, natural morning light.',
        'Bento box label design, clean packaging mockup; headline "手作り弁当"; small ingredient list: "鮭・卵焼き・梅干し"; price sticker "650円"; modern sans-serif Japanese fonts.',
        'Japanese train platform signage, realistic JR-style; overhead sign with station name "新宿駅" and platform number "3番線"; add direction arrow and destination "快速 高尾行"; crisp transport typography.',
        'Ukiyo-e inspired poster, textured washi paper; bold brushstroke kanji "旅" dominating the composition; red seal mark "江戸風"; muted indigo palette.',
        'Coffee cup + sleeve mockup on café counter; sleeve text "本日のおすすめ" and below it "深煎りブレンド"; chalkboard menu bokeh background; cozy light, photoreal.',
        'Smartphone weather app UI screen; header text "今日の天気"; weekday labels in Japanese: "月", "火", "水", "木", "金"; include the condition tag "晴れ"; temperature readouts; sleek flat design.',
        'Dramatic sci-fi movie poster; title "最後の光：遥かなる記憶の果て" in large metallic Japanese type; tagline "希望は消えない" below; cinematic grading, star field background.',  # noqa: RUF001
        'Classroom chalkboard, dusty chalk texture; handwritten announcements: "テストは金曜日" and "がんばろう！"; eraser marks and doodles; warm afternoon light.',  # noqa: RUF001
        'Summer festival poster, bold typography; headline "夏祭り"; date "8月15日" and location "中央公園"; lantern graphics; vibrant, printable A3 layout.',
        'City safety billboard near crosswalk; big, high-contrast warnings "安全第一" and "スピード注意"; reflective materials, urban street scene, twilight.',
        'Hanging calligraphy scroll (kakejiku), tatami room; vertical brush poem "静けさの中に光あり、心は波のようにおだやかに満ち、ひとすじの風が時を運ぶ"; red artist seal "青風"; soft morning light and paper texture.',
        'Sushi bar handwritten menu board; chalk marker items: "にぎり 5貫 800円", line items "まぐろ", "サーモン", "えび"; rustic wooden frame; warm tungsten lighting.',
        'Hotel door hanger set (front/back); side A text "清掃をお願いします"; side B text "起こさないでください"; clean layout with Japanese text prominent.',
        'Futuristic holographic ad in a rainy city; luminous headline "未来都市"; subline "今すぐ登録"; floating UI elements and scan-grid motifs; glossy reflections.',
        'Travel postcard collage, Kyoto landmarks; stamped welcome text "ようこそ京都"; handwritten note "また来ます" in pen; vintage grain, off-white paper.',
    ]

    filename_prefix = f"qwenimage_{width}_{height}_{mesh_device.shape[0]}x{mesh_device.shape[1]}"
    if not traced:
        filename_prefix += "_untraced"

    def run(*, prompt: str, number: int, seed: int) -> None:
        images = pipeline(
            prompts=[prompt],
            num_inference_steps=50,
            seed=seed,
            width=width,
            height=height,
            traced=traced,
        )

        output_filename = f"{filename_prefix}_{number}.png"
        images[0].save(output_filename)
        logger.info(f"Image saved as {output_filename}")

    if no_prompt:
        for i, prompt in enumerate(prompts[:1]):  # only run with the first prompt by default. Increase as needed.
            run(prompt=prompt, number=i, seed=0)
    else:
        prompt = prompts[0]
        for i in itertools.count():
            new_prompt = input("Enter the input prompt, or q to exit: ")
            if new_prompt:
                prompt = new_prompt
            if prompt[0] == "q":
                break
            run(prompt=prompt, number=i, seed=i)


@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_DEVICE_PARAMS, indirect=True)
def test_qwenimage_inpaint(*, mesh_device: ttnn.MeshDevice, is_ci_env: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    if is_ci_env:
        monkeypatch.setenv("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")

    pipeline = QwenImagePipeline.create_pipeline(mesh_device=mesh_device)

    image = Image.open(DOG_IMAGE_PATH).convert("RGB")
    # Repaints the dog, with a margin around it.
    mask = Image.new("L", (512, 512), 0)
    mask.paste(255, (130, 20, 430, 425))
    (inpainted,) = pipeline(
        prompts=["A ginger cat standing on a skateboard."],
        num_inference_steps=50,
        seed=0,
        image=image,
        mask_image=mask,
    )

    image.save("qwenimage_inpaint_input.png")
    inpainted.save("qwenimage_inpaint_output.png")

    before = np.asarray(image, dtype=np.float32)
    after = np.asarray(inpainted.resize(image.size), dtype=np.float32)
    inside = np.asarray(mask) > 0
    # Masks are blended at latent resolution, so the band along the mask edge may change too.
    edge = np.asarray(mask.filter(ImageFilter.MaxFilter(33))) > 0
    outside_diff = np.abs(after - before)[~edge].mean()
    inside_diff = np.abs(after - before)[inside].mean()
    logger.info(f"mean absolute change outside the mask {outside_diff:.2f}, inside {inside_diff:.2f}")
    assert outside_diff < 5
    # Some of the box stays background, so only the change relative to the outside is meaningful.
    assert inside_diff > 3 * outside_diff


EDIT_CHECKPOINTS = [
    pytest.param("Qwen/Qwen-Image-Edit", id="edit"),
    pytest.param("Qwen/Qwen-Image-Edit-2509", id="2509"),
    pytest.param("Qwen/Qwen-Image-Edit-2511", id="2511"),
]


@pytest.mark.parametrize(("mesh_device", "device_params"), EDIT_MESH_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("checkpoint_name", EDIT_CHECKPOINTS)
def test_qwenimage_edit(
    *, mesh_device: ttnn.MeshDevice, checkpoint_name: str, is_ci_env: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if is_ci_env:
        monkeypatch.setenv("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")

    pipeline = QwenImagePipeline.create_pipeline(mesh_device=mesh_device, checkpoint_name=checkpoint_name)

    image = Image.open(DOG_IMAGE_PATH).convert("RGB")
    (edited,) = pipeline(
        prompts=["Make the dog wear a red knitted scarf."], num_inference_steps=50, seed=0, image=image
    )

    tag = checkpoint_name.rsplit("/", 1)[-1].lower()
    image.save(f"qwenimage_{tag}_input.png")
    edited.save(f"qwenimage_{tag}_output.png")

    # The dog image is square, so the output is 1024².
    assert edited.size == (1024, 1024)
    before = np.asarray(image.resize(edited.size), dtype=np.float32)
    after = np.asarray(edited, dtype=np.float32)
    diff = np.abs(after - before).mean()
    logger.info(f"mean absolute change {diff:.2f}")
    # An edit keeps the composition and changes some of it.
    assert 1 < diff < 40


@pytest.mark.parametrize(("mesh_device", "device_params"), EDIT_MESH_DEVICE_PARAMS, indirect=True)
def test_qwenimage_edit_multiple_images(
    *, mesh_device: ttnn.MeshDevice, is_ci_env: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Edit-2509 combines two input images."""
    if is_ci_env:
        monkeypatch.setenv("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")

    pipeline = QwenImagePipeline.create_pipeline(mesh_device=mesh_device, checkpoint_name="Qwen/Qwen-Image-Edit-2509")

    dog = Image.open(DOG_IMAGE_PATH).convert("RGB")
    scarf = Image.new("RGB", (512, 512), (200, 30, 30))
    (edited,) = pipeline(
        prompts=["The dog in Picture 1 wears a scarf in the color of Picture 2."],
        num_inference_steps=50,
        seed=0,
        image=[dog, scarf],
    )
    edited.save("qwenimage_edit_2509_two_images_output.png")
    # The output takes the size of the last image.
    assert edited.size == (1024, 1024)


@pytest.mark.parametrize(("mesh_device", "device_params"), EDIT_MESH_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("checkpoint_name", EDIT_CHECKPOINTS)
def test_qwenimage_edit_inpaint(
    *, mesh_device: ttnn.MeshDevice, checkpoint_name: str, is_ci_env: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if is_ci_env:
        monkeypatch.setenv("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")

    pipeline = QwenImagePipeline.create_pipeline(mesh_device=mesh_device, checkpoint_name=checkpoint_name)

    image = Image.open(DOG_IMAGE_PATH).convert("RGB")
    mask = Image.new("L", image.size, 0)
    w, h = image.size
    mask.paste(255, (int(0.25 * w), int(0.04 * h), int(0.84 * w), int(0.83 * h)))
    (edited,) = pipeline(
        prompts=["Turn the dog into a ginger cat."],
        num_inference_steps=50,
        seed=0,
        image=image,
        mask_image=mask,
        strength=1.0,
    )
    tag = checkpoint_name.rsplit("/", 1)[-1].lower()
    edited.save(f"qwenimage_{tag}_inpaint_output.png")

    before = np.asarray(image.resize(edited.size), dtype=np.float32)
    after = np.asarray(edited, dtype=np.float32)
    mask = mask.resize(edited.size)
    inside = np.asarray(mask) > 0
    edge = np.asarray(mask.filter(ImageFilter.MaxFilter(65))) > 0
    outside_diff = np.abs(after - before)[~edge].mean()
    inside_diff = np.abs(after - before)[inside].mean()
    logger.info(f"mean absolute change outside the mask {outside_diff:.2f}, inside {inside_diff:.2f}")
    assert outside_diff < 5
    assert inside_diff > 3 * outside_diff
