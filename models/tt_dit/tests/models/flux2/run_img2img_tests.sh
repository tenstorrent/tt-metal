#!/usr/bin/env bash
# Run Flux2 img2img tests with a persistent TT-Metal DiT weight cache.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

export TT_DIT_CACHE_DIR="/proj_sw/user_dev/${USER}/tt_dit_cache"

cd "${REPO_ROOT}"
# shellcheck source=/dev/null
source python_env/bin/activate

STAGE="${1:-all}"

run_cpu() {
    pytest models/tt_dit/tests/models/flux2/test_transformer_flux2.py::test_img2img_latent_helpers \
        models/tt_dit/tests/models/flux2/test_vae_flux2.py::test_vae_encode_img2img \
        models/tt_dit/tests/models/flux2/test_prompt_encoder.py::test_format_input_i2i \
        models/tt_dit/tests/models/flux2/test_img2img_flux2.py::test_img2img_spatial_rope_ids \
        models/tt_dit/tests/models/flux2/test_img2img_flux2.py::test_img2img_euler_step_matches_torch -v
}

run_combine() {
    pytest models/tt_dit/tests/models/flux2/test_img2img_flux2.py::test_combine_img2img_spatial_input \
        models/tt_dit/tests/models/flux2/test_img2img_flux2.py::test_extract_noise_latents_from_combined -v
}

run_transformer() {
    pytest models/tt_dit/tests/models/flux2/test_transformer_flux2.py::test_transformer_img2img \
        models/tt_dit/tests/models/flux2/test_img2img_flux2.py::test_transformer_img2img_separate_latents -v
}

run_prompt_encoder() {
    pytest models/tt_dit/tests/models/flux2/test_prompt_encoder.py::test_upsample_i2i -v
}

run_pipeline() {
    pytest models/tt_dit/tests/models/flux2/test_pipeline_flux2.py::test_pipeline_img2img -v
}

echo "Using TT_DIT_CACHE_DIR=${TT_DIT_CACHE_DIR}"

case "${STAGE}" in
    cpu) run_cpu ;;
    transformer) run_transformer ;;
    prompt_encoder) run_prompt_encoder ;;
    pipeline) run_pipeline ;;
    device)
        run_combine
        run_transformer
        run_prompt_encoder
        run_pipeline
        ;;
    all)
        run_cpu
        run_combine
        run_transformer
        run_prompt_encoder
        run_pipeline
        ;;
    *)
        echo "Usage: $0 [cpu|combine|transformer|prompt_encoder|pipeline|device|all]" >&2
        exit 1
        ;;
esac
