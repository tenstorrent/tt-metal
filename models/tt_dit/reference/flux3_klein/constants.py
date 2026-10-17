# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Image Klein constants for host code (packing, scheduler, tests) that must not import torch or transformers.

Every value is copied from https://github.com/black-forest-labs/flux-3-image-klein at e53f4ea; the upstream file and
line are cited next to each one. The vendored modules beside this file keep their own, identical, definitions verbatim
so that their diff against upstream stays small; the reference test for this model should pin the two in sync.
"""

# --- src/flux3/dit.py ---------------------------------------------------------------------------------------------
MODES = ("image", "image_cond", "text")  # dit.py:11, sorted; the order tokens take inside a sample
IN_CHANNELS = {"image": 128, "image_cond": 128, "text": 20480}  # dit.py:12
HIDDEN = 3072  # dit.py:13
HEADS = 24  # dit.py:14
DEPTH = 5  # dit.py:15, per-mode blocks (image, image_cond and text stacks)
DEPTH_SINGLE = 28  # dit.py:16, shared single-stream blocks over [image | image_cond | text]
AXES_DIM = (32, 32, 32, 32)  # dit.py:17, RoPE dims of the (t, h, w, l) id axes
MLP_RATIO = 3.0  # dit.py:18
THETA = 10000  # dit.py:19
HEAD_DIM = HIDDEN // HEADS  # dit.py:143 (QKNorm(hidden // heads)); equals sum(AXES_DIM)
TIME_EMBED_DIM = 256  # dit.py:36 (timestep_embedding default) and :162, :164 (MLPEmbedder(256, HIDDEN))

# --- src/flux3/autoencoder.py -------------------------------------------------------------------------------------
DOWNSAMPLING_RATIO = 16  # autoencoder.py:13, 8x VAE + 2x2 patchify
LATENT_CHANNELS = 128  # autoencoder.py:14

# --- src/flux3/text_encoder.py ------------------------------------------------------------------------------------
OUTPUT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)  # text_encoder.py:9, Qwen3-VL hidden_states taps; 8 x 2560 = 20480
MAX_LENGTH = 4096  # text_encoder.py:10
PAD_MULTIPLE = 80  # text_encoder.py:11, text length rounded up to a multiple of 80 (capped at MAX_LENGTH)

# --- src/flux3/sampling.py and src/flux3/pipeline.py --------------------------------------------------------------
ALPHA = 6.93  # pipeline.py:63 (Pipeline.generate default), used by sampling.py:9-11 timesteps(steps, alpha)
IMAGE_ROPE_SCALE = 0.5  # pipeline.py:24; scaled_ids (pipeline.py:123-126) scales the (h, w) axes of image ids
DEFAULT_STEPS = {"base": 50, "distilled": 16}  # pipeline.py:20-21, MODELS[...]["steps"]
DEFAULT_GUIDANCE = {
    "base": 3.5,
    "distilled": None,
}  # pipeline.py:20-21, MODELS[...]["guidance"]; None = flow map, no CFG

# --- src/flux3/checkpoint.py --------------------------------------------------------------------------------------
FLOW_MAP_KEY = "ref_time_in.in_layer.weight"  # checkpoint.py:13; present in the distilled (flow-map) checkpoint only
