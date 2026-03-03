# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import pytest
from loguru import logger

import ttnn

from ....pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 47000000}],
    indirect=True,
)
@pytest.mark.parametrize(("width", "height", "num_inference_steps"), [(1024, 1024, 50)])
@pytest.mark.parametrize(
    "mesh_device, cfg, sp, tp, encoder_tp, vae_tp, topology, num_links",
    [
        # 2x4 config with sp enabled - sp on axis 0 enables fsdp weight sharding (no cfg parallel)
        [(2, 4), (2, 0), (1, 0), (4, 1), (4, 1), (4, 1), ttnn.Topology.Linear, 1],
        [(4, 8), (2, 1), (4, 0), (4, 1), (4, 1), (4, 1), ttnn.Topology.Linear, 4],
    ],
    ids=[
        "2x4sp1tp4",
        "4x8sp4tp4",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "use_torch_text_encoder",
    [
        # pytest.param(True, id="encoder_cpu"),
        pytest.param(False, id="encoder_device"),
    ],
)
@pytest.mark.parametrize(
    "traced",
    [
        pytest.param(True, id="traced"),
        pytest.param(False, id="not_traced"),
    ],
)
def test_qwenimage_pipeline(
    *,
    mesh_device: ttnn.MeshDevice,
    width: int,
    height: int,
    num_inference_steps: int,
    cfg: tuple[int, int],
    sp: tuple[int, int],
    tp: tuple[int, int],
    encoder_tp: tuple[int, int],
    vae_tp: tuple[int, int],
    topology: ttnn.Topology,
    num_links: int,
    no_prompt: bool,
    use_torch_text_encoder: bool,
    traced: bool,
    is_ci_env: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Set TT_DIT_CACHE in CI environment. The path is needed for dynamic load when necessary.
    if is_ci_env:
        monkeypatch.setenv("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")

    pipeline = QwenImagePipeline.create_pipeline(
        mesh_device=mesh_device,
        dit_cfg=cfg,
        dit_sp=sp,
        dit_tp=tp,
        encoder_tp=encoder_tp,
        vae_tp=vae_tp,
        use_torch_text_encoder=use_torch_text_encoder,
        use_torch_vae_decoder=False,
        num_links=num_links,
        topology=topology,
        width=width,
        height=height,
    )

    prompts = [
        'A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light '
        'beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the '
        'poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition'
        ", Ultra HD, 4K, cinematic composition."
        'Tokyo neon alley at night, rain-slick pavement, cinematic cyberpunk lighting; include glowing sign text "深夜営業" in bold neon above a doorway; moody reflections, shallow depth of field.',
        'Steamy ramen shop entrance at dusk; fabric noren curtain gently swaying; print "しょうゆラーメン" across the curtain in thick brush-style kana; warm lantern light, photorealistic.',
        'Minimalist tea poster, cream background, elegant layout; vertical calligraphy "抹茶" centered in sumi ink; small red hanko-style seal "本格" in the corner; high-resolution graphic design.',
        'Hardcover fantasy novel cover, textured paper, gold foil; title text "物語のはじまり" centered; author line "山本ひかり" below; tasteful serif typography, dramatic vignette illustration.',
        'Anime manga panel, dynamic action lines; speech bubble with clear Japanese text "大丈夫、行こう！"; bold hand-lettered style; black-and-white screentone shading.',
        'Shinto shrine ema (wooden wish plaque) close-up; handwritten ink message "合格祈願"; tied with a red cord; shallow depth of field, natural morning light.',
        'Bento box label design, clean packaging mockup; headline "手作り弁当"; small ingredient list: "鮭・卵焼き・梅干し"; price sticker "650円"; modern sans-serif Japanese fonts.',
        'Japanese train platform signage, realistic JR-style; overhead sign with station name "新宿駅" and platform number "3番線"; add direction arrow and destination "快速 高尾行"; crisp transport typography.',
        'Ukiyo-e inspired poster, textured washi paper; bold brushstroke kanji "旅" dominating the composition; red seal mark "江戸風"; muted indigo palette.',
        'Coffee cup + sleeve mockup on café counter; sleeve text "本日のおすすめ" and below it "深煎りブレンド"; chalkboard menu bokeh background; cozy light, photoreal.',
        'Smartphone weather app UI screen; header text "今日の天気"; weekday labels in Japanese: "月", "火", "水", "木", "金"; include the condition tag "晴れ"; temperature readouts; sleek flat design.',
        'Dramatic sci-fi movie poster; title "最後の光：遥かなる記憶の果て" in large metallic Japanese type; tagline "希望は消えない" below; cinematic grading, star field background.',
        'Classroom chalkboard, dusty chalk texture; handwritten announcements: "テストは金曜日" and "がんばろう！"; eraser marks and doodles; warm afternoon light.',
        'Summer festival poster, bold typography; headline "夏祭り"; date "8月15日" and location "中央公園"; lantern graphics; vibrant, printable A3 layout.',
        'City safety billboard near crosswalk; big, high-contrast warnings "安全第一" and "スピード注意"; reflective materials, urban street scene, twilight.',
        'Hanging calligraphy scroll (kakejiku), tatami room; vertical brush poem "静けさの中に光あり、心は波のようにおだやかに満ち、ひとすじの風が時を運ぶ"; red artist seal "青風"; soft morning light and paper texture.',
        'Sushi bar handwritten menu board; chalk marker items: "にぎり 5貫 800円", line items "まぐろ", "サーモン", "えび"; rustic wooden frame; warm tungsten lighting.',
        'Hotel door hanger set (front/back); side A text "清掃をお願いします"; side B text "起こさないでください"; clean layout with Japanese text prominent.',
        'Futuristic holographic ad in a rainy city; luminous headline "未来都市"; subline "今すぐ登録"; floating UI elements and scan-grid motifs; glossy reflections.',
        'Travel postcard collage, Kyoto landmarks; stamped welcome text "ようこそ京都"; handwritten note "また来ます" in pen; vintage grain, off-white paper.',
    ]

    _, cfg_axis = cfg
    _, sp_axis = sp
    _, tp_axis = tp
    mesh_test_id = f"{mesh_device.shape[0]}x{mesh_device.shape[1]}cfg{cfg_axis}sp{sp_axis}tp{tp_axis}"
    filename_prefix = f"qwenimage_{width}_{height}_{mesh_test_id}"
    if use_torch_text_encoder:
        filename_prefix += "_encodercpu"
    if not traced:
        filename_prefix += "_untraced"

    def run(*, prompt: str, number: int, seed: int) -> None:
        images = pipeline(
            prompts=[prompt],
            negative_prompts=[None],
            num_inference_steps=num_inference_steps,
            cfg_scale=4.0,
            seed=seed,
            traced=traced,
            vae_traced=False,
            encoder_traced=False,
        )

        output_filename = f"{filename_prefix}_{number}.png"
        images[0].save(output_filename)
        logger.info(f"Image saved as {output_filename}")

    if no_prompt:
        for i, prompt in enumerate(prompts[:2]):  # only run with the first prompt by default. Increase as needed.
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
