# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Prompt enhancement for the LTX-2 pipelines.

LTX-2 is trained on long, caption-style prompts. At serving time the raw user prompt is rewritten
into that style by a small instruction-tuned LLM (Gemma 4 E2B) before it reaches the Gemma-3 text
encoder. This module is that string→string stage. The encoder path is untouched: the pipelines
rewrite first, then hash and encode the rewritten text exactly as they would a raw one.
"""

from __future__ import annotations

import contextlib
import gc
import json
import math
import os
import time
from abc import ABC, abstractmethod
from typing import Literal

import torch
from loguru import logger

EnhanceMode = Literal["t2v", "i2v"]

DEFAULT_ENHANCER_PATH = "google/gemma-4-E2B-it"

# Generation settings of the reference enhancer (LTX-2 ``ltx_core``; mirrored by diffusers'
# ``LTX2Pipeline.enhance_prompt``).
ENHANCER_MAX_NEW_TOKENS = 512
ENHANCER_TEMPERATURE = 0.7
ENHANCER_SEED = 10
# The checkpoint's generation_config nucleus settings. Both backends pass them explicitly so neither
# depends on a library default (``sample_host`` defaults to top_p=0.08) and same-seed A/B stays like-for-like.
ENHANCER_TOP_K = 64
ENHANCER_TOP_P = 0.95


class PromptTooLongError(ValueError):
    """The templated prompt does not fit the rewriter's context next to its token budget."""


def default_ltx_enhancer() -> str:
    return os.environ.get("LTX_ENHANCER_PATH") or DEFAULT_ENHANCER_PATH


# System prompts verbatim from LTX-2 (Apache-2.0), including the leading newline:
# https://github.com/Lightricks/LTX-2/tree/ae855f8538843825f9015a419cf4ba5edaf5eec2/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts
T2V_SYSTEM_PROMPT = """
You are a Creative Assistant. Given a user's raw input prompt describing a scene or concept, expand it into a detailed
video generation prompt with specific visuals and integrated audio to guide a text-to-video model.

#### Guidelines
- Strictly follow all aspects of the user's raw input: include every element requested (style, visuals, motions,
  actions, camera movement, audio).
    - If the input is vague, invent concrete details: lighting, textures, materials, scene settings, etc.
        - For characters: describe gender, clothing, hair, expressions. DO NOT invent unrequested characters.
- Use active language: present-progressive verbs ("is walking," "speaking"). If no action specified, describe natural
  movements.
- Maintain chronological flow: use temporal connectors ("as," "then," "while").
- Audio layer: Describe complete soundscape (background audio, ambient sounds, SFX, speech/music when requested).
  Integrate sounds chronologically alongside actions. Be specific (e.g., "soft footsteps on tile"), not vague (e.g.,
  "ambient sound is present").
- Speech (only when requested):
    - For ANY speech-related input (talking, conversation, singing, etc.), ALWAYS include exact words in quotes with
      voice characteristics (e.g., "The man says in an excited voice: 'You won't believe what I just saw!'").
    - Specify language if not English and accent if relevant.
- Style: Include visual style at the beginning: "Style: <style>, <rest of prompt>." Default to cinematic-realistic if
  unspecified. Omit if unclear.
- Visual and audio only: NO non-visual/auditory senses (smell, taste, touch).
- Restrained language: Avoid dramatic/exaggerated terms. Use mild, natural phrasing.
    - Colors: Use plain terms ("red dress"), not intensified ("vibrant blue," "bright red").
    - Lighting: Use neutral descriptions ("soft overhead light"), not harsh ("blinding light").
    - Facial features: Use delicate modifiers for subtle features (i.e., "subtle freckles").

#### Important notes:
- Analyze the user's raw input carefully. In cases of FPV or POV, exclude the description of the subject whose POV is
  requested.
- Camera motion: DO NOT invent camera motion unless requested by the user.
- Speech: DO NOT modify user-provided character dialogue unless it's a typo.
- No timestamps or cuts: DO NOT use timestamps or describe scene cuts unless explicitly requested.
- Format: DO NOT use phrases like "The scene opens with...". Start directly with Style (optional) and chronological
  scene description.
- Format: DO NOT start your response with special characters.
- DO NOT invent dialogue unless the user mentions speech/talking/singing/conversation.
- If the user's raw input prompt is highly detailed, chronological and in the requested format: DO NOT make major edits
  or introduce new elements. Add/enhance audio descriptions if missing.

#### Output Format (Strict):
- Single continuous paragraph in natural language (English).
- NO titles, headings, prefaces, code fences, or Markdown.
- If unsafe/invalid, return original user prompt. Never ask questions or clarifications.

Your output quality is CRITICAL. Generate visually rich, dynamic prompts with integrated audio for high-quality video
generation.

#### Example Input: "A woman at a coffee shop talking on the phone" Output: Style: realistic with cinematic lighting.
In a medium close-up, a woman in her early 30s with shoulder-length brown hair sits at a small wooden table by the
window. She wears a cream-colored turtleneck sweater, holding a white ceramic coffee cup in one hand and a smartphone
to her ear with the other. Ambient cafe sounds fill the space—espresso machine hiss, quiet conversations, gentle
clinking of cups. The woman listens intently, nodding slightly, then takes a sip of her coffee and sets it down with a
soft clink. Her face brightens into a warm smile as she speaks in a clear, friendly voice, 'That sounds perfect! I'd
love to meet up this weekend. How about Saturday afternoon?' She laughs softly—a genuine chuckle—and shifts in her
chair. Behind her, other patrons move subtly in and out of focus. 'Great, I'll see you then,' she concludes cheerfully,
lowering the phone.
"""

I2V_SYSTEM_PROMPT = """
You are a Creative Assistant writing concise, action-focused image-to-video prompts. Given an image (first frame) and
user Raw Input Prompt, generate a prompt to guide video generation from that image.

#### Guidelines:
- Analyze the Image: Identify Subject, Setting, Elements, Style and Mood.
- Follow user Raw Input Prompt: Include all requested motion, actions, camera movements, audio, and details. If in
  conflict with the image, prioritize user request while maintaining visual consistency (describe transition from image
  to user's scene).
- Describe only changes from the image: Don't reiterate established visual details. Inaccurate descriptions may cause
  scene cuts.
- Active language: Use present-progressive verbs ("is walking," "speaking"). If no action specified, describe natural
  movements.
- Chronological flow: Use temporal connectors ("as," "then," "while").
- Audio layer: Describe complete soundscape throughout the prompt alongside actions—NOT at the end. Align audio
  intensity with action tempo. Include natural background audio, ambient sounds, effects, speech or music (when
  requested). Be specific (e.g., "soft footsteps on tile") not vague (e.g., "ambient sound").
- Speech (only when requested): Provide exact words in quotes with character's visual/voice characteristics (e.g., "The
  tall man speaks in a low, gravelly voice"), language if not English and accent if relevant. If general conversation
  mentioned without text, generate contextual quoted dialogue. (i.e., "The man is talking" input -> the output should
  include exact spoken words, like: "The man is talking in an excited voice saying: 'You won't believe what I just
  saw!' His hands gesture expressively as he speaks, eyebrows raised with enthusiasm. The ambient sound of a quiet room
  underscores his animated speech.")
- Style: Include visual style at beginning: "Style: <style>, <rest of prompt>." If unclear, omit to avoid conflicts.
- Visual and audio only: Describe only what is seen and heard. NO smell, taste, or tactile sensations.
- Restrained language: Avoid dramatic terms. Use mild, natural, understated phrasing.

#### Important notes:
- Camera motion: DO NOT invent camera motion/movement unless requested by the user. Make sure to include camera motion
  only if specified in the input.
- Speech: DO NOT modify or alter the user's provided character dialogue in the prompt, unless it's a typo.
- No timestamps or cuts: DO NOT use timestamps or describe scene cuts unless explicitly requested.
- Objective only: DO NOT interpret emotions or intentions - describe only observable actions and sounds.
- Format: DO NOT use phrases like "The scene opens with..." / "The video starts...". Start directly with Style
  (optional) and chronological scene description.
- Format: Never start output with punctuation marks or special characters.
- DO NOT invent dialogue unless the user mentions speech/talking/singing/conversation.
- Your performance is CRITICAL. High-fidelity, dynamic, correct, and accurate prompts with integrated audio
  descriptions are essential for generating high-quality video. Your goal is flawless execution of these rules.

#### Output Format (Strict):
- Single concise paragraph in natural English. NO titles, headings, prefaces, sections, code fences, or Markdown.
- If unsafe/invalid, return original user prompt. Never ask questions or clarifications.

#### Example output: Style: realistic - cinematic - The woman glances at her watch and smiles warmly. She speaks in a
cheerful, friendly voice, "I think we're right on time!" In the background, a café barista prepares drinks at the
counter. The barista calls out in a clear, upbeat tone, "Two cappuccinos ready!" The sound of the espresso machine
hissing softly blends with gentle background chatter and the light clinking of cups on saucers.
"""


def system_prompt_for(mode: EnhanceMode) -> str:
    if mode == "t2v":
        return T2V_SYSTEM_PROMPT
    if mode == "i2v":
        return I2V_SYSTEM_PROMPT
    raise ValueError(f"unknown enhance mode {mode!r}; expected 't2v' or 'i2v'")


def build_messages(prompt: str, mode: EnhanceMode = "t2v") -> list[dict[str, str]]:
    """Chat turns for the rewriter. The ``user prompt:`` prefix is part of the reference input format."""
    return [
        {"role": "system", "content": system_prompt_for(mode)},
        {"role": "user", "content": f"user prompt: {prompt}"},
    ]


class PromptEnhancer(ABC):
    """Rewrites a raw user prompt into LTX caption style.

    Implementations own their model lifetime. The pipeline calls ``ensure_loaded`` before the first
    ``enhance`` and never caches rewrites itself: sampling makes a rewrite a function of
    ``(prompt, seed)``, and the embedding cache downstream is keyed on the rewritten text."""

    # Bounds the rewrite so it fits the encoder's fixed token window; the pipeline asserts against it.
    max_new_tokens: int = ENHANCER_MAX_NEW_TOKENS

    @property
    @abstractmethod
    def name(self) -> str:
        """Short backend/model label for logs and timing rows."""

    @abstractmethod
    def ensure_loaded(self) -> None:
        """Load the rewriter if it is not resident. Idempotent."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the rewriter is resident (weights allocated) right now."""

    def warmup(self) -> None:
        """Load and compile whatever the first ``enhance`` would otherwise pay for. Default: just load."""
        self.ensure_loaded()

    def unload(self) -> None:
        """Release the rewriter's weights. Default: nothing resident to release."""

    @abstractmethod
    def enhance(
        self,
        prompt: str,
        *,
        mode: EnhanceMode = "t2v",
        image_path: str | None = None,
        seed: int = ENHANCER_SEED,
    ) -> str:
        """Return the rewritten prompt. ``image_path`` is the I2V conditioning frame when there is one."""


class HostPromptEnhancer(PromptEnhancer):
    """Runs the rewriter on the host CPU through ``transformers``.

    Text-only: an I2V conditioning image is not shown to the model, so ``mode="i2v"`` applies the I2V
    system prompt to the text alone. Throughput is a few tokens per second on CPU, so a full
    512-token rewrite costs on the order of a minute or two; this backend is for bring-up and for
    A/B-ing checkpoints, not for serving."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        max_new_tokens: int = ENHANCER_MAX_NEW_TOKENS,
        temperature: float = ENHANCER_TEMPERATURE,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_path = model_path or default_ltx_enhancer()
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.dtype = dtype
        self._tokenizer = None
        self._model = None
        self._warned_image = False

    @property
    def name(self) -> str:
        return f"host:{self.model_path}"

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.time()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path, dtype=self.dtype)
        self._model.eval()
        logger.info(f"prompt enhancer loaded on host: {self.model_path} ({time.time() - t0:.1f}s)")

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None

    def enhance(
        self,
        prompt: str,
        *,
        mode: EnhanceMode = "t2v",
        image_path: str | None = None,
        seed: int = ENHANCER_SEED,
    ) -> str:
        self.ensure_loaded()
        if image_path is not None and not self._warned_image:
            logger.warning("host prompt enhancer is text-only; the I2V conditioning image is not shown to the rewriter")
            self._warned_image = True

        text = self._tokenizer.apply_chat_template(
            build_messages(prompt, mode), tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt")

        # Reproducibility follows the reference: seed torch's global RNG rather than plumbing a Generator,
        # which ``transformers`` ``generate`` does not accept. Greedy decoding draws nothing, so it leaves
        # the RNG alone (as the device backend does).
        if self.temperature > 0:
            torch.manual_seed(seed)
            sampling = {
                "do_sample": True,
                "temperature": self.temperature,
                "top_k": ENHANCER_TOP_K,
                "top_p": ENHANCER_TOP_P,
            }
        else:
            sampling = {"do_sample": False}
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, **sampling)
        new_tokens = out[0, inputs.input_ids.shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# --- device backend ---------------------------------------------------------------------------

ENHANCER_MAX_SEQ_LEN = 2048
# gemma4 paged attention block; the page table is sized to max_seq_len for batch 1.
ENHANCER_PAGE_BLOCK_SIZE = 64
# Gemma-4 generation_config eos ids: <eos>, <|tool_response>, <turn|>. The generator's own stop set is
# only [<eos>], and an instruct reply closes with <turn|>, so without these a rewrite never stops.
GEMMA4_EOS_IDS = (1, 50, 106)
SHARED_ENHANCER_CACHE_DIR = "/mnt/models/huggingface/tt_cache/gemma-4-E2B-it"
USER_ENHANCER_CACHE_DIR = "~/.cache/tt-gemma4-e2b"
# Prefill compiles once per padded length bucket (<=1024 -> 1024, else the next power of two). The T2V
# system prompt alone is ~975 Gemma-4 tokens, so a short user prompt sits in the 1024 bucket and anything
# past ~50 user tokens (every stock LTX prompt) in the 2048 one; one warm prompt per bucket compiles both.
WARMUP_PROMPTS = (
    "a cat on a windowsill",
    "a woman in a grey wool coat is walking along a rain-wet harbour promenade at dusk, holding a paper cup "
    "of coffee in both hands. Fishing boats rock gently at their moorings behind her, their hulls painted red "
    "and white, and strings of warm bulbs sway between the lamp posts. She pauses at the railing, looks out at "
    "the grey water, and takes a slow sip as a gull lands on the post beside her. Gentle waves lap against the "
    "stone quay, rigging clinks against the masts, and distant voices and the low hum of a diesel engine drift "
    "across the water as the gull calls out once and lifts off again",
)
WARMUP_NEW_TOKENS = 8


def resolve_enhancer_cache_dir(cache_dir: str | None = None) -> str:
    """Directory for the rewriter's converted weight cache: the argument, else ``LTX_ENHANCER_CACHE_DIR``,
    else the shared tt_cache dir when it exists and is writable, else a per-user dir."""
    if cache_dir:
        return os.path.expanduser(cache_dir)
    from_env = os.environ.get("LTX_ENHANCER_CACHE_DIR")
    if from_env:
        return os.path.expanduser(from_env)
    if os.path.isdir(SHARED_ENHANCER_CACHE_DIR) and os.access(SHARED_ENHANCER_CACHE_DIR, os.W_OK):
        return SHARED_ENHANCER_CACHE_DIR
    return os.path.expanduser(USER_ENHANCER_CACHE_DIR)


def bind_prompt_enhancer_mesh(enhancer: PromptEnhancer | None, mesh_device, *, ccl_topology=None) -> bool:
    """Give a device enhancer built without a mesh the pipeline's own handle and CCL topology. Returns
    True when it bound.

    Only an unbound enhancer is touched; one already holding a different handle is left alone with a
    warning, since the pipeline cannot know whether that handle aliases its submesh. The topology rides
    along because gemma4 picks its own default (Ring on any 8+ chip Blackhole mesh), which is wrong on
    a handle whose fabric is a line."""
    if enhancer is None or not hasattr(enhancer, "mesh_device"):
        return False
    if enhancer.mesh_device is None:
        enhancer.mesh_device = mesh_device
        if ccl_topology is not None and getattr(enhancer, "ccl_topology", None) is None:
            enhancer.ccl_topology = ccl_topology
        return True
    if enhancer.mesh_device is not mesh_device:
        logger.warning(
            f"prompt enhancer {enhancer.name} holds a mesh handle that is not the pipeline's; "
            "overlapping handles share no allocator and corrupt each other"
        )
    return False


def topology_env_value(topology) -> str:
    """``GEMMA4_CCL_TOPOLOGY`` spelling of a ``ttnn.Topology`` (compared by name, so this module needs no ttnn)."""
    name = str(topology).rsplit(".", 1)[-1].lower()
    if name not in ("linear", "ring"):
        raise ValueError(f"unsupported CCL topology for the prompt enhancer: {topology!r}")
    return name


@contextlib.contextmanager
def _scoped_env(name: str, value: str):
    """Set ``name`` for the duration of the block and restore whatever it was before (unset included)."""
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def resolve_model_dir(model_path: str) -> str:
    """A local checkpoint directory for ``model_path``: the path itself when it is one, else the hub cache's
    snapshot of that repo id, else the id unchanged. gemma4 reads safetensors straight out of a directory but
    instantiates the whole HF model on host for a bare id, so the snapshot is the loader to prefer."""
    if os.path.isdir(model_path):
        return model_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_path, local_files_only=True)
    except Exception as e:  # noqa: BLE001 — no cached snapshot: gemma4 loads the id through transformers
        logger.info(f"prompt enhancer: no local snapshot for {model_path!r} ({type(e).__name__}); loading by id")
        return model_path


def resolve_stop_tokens(model_path: str, tokenizer) -> list[int]:
    """The generator's stop set plus the checkpoint's ``generation_config.json`` eos ids (``GEMMA4_EOS_IDS``
    when no local copy of that file is reachable)."""
    stops = {int(t) for t in (getattr(tokenizer, "stop_tokens", None) or [])}
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        stops.add(int(eos))
    eos_ids = _read_generation_config_eos(model_path)
    if eos_ids is None:
        eos_ids = GEMMA4_EOS_IDS
    stops.update(int(i) for i in eos_ids)
    return sorted(stops)


def _read_generation_config_eos(model_path: str) -> list[int] | None:
    path = None
    if os.path.isdir(model_path):
        candidate = os.path.join(model_path, "generation_config.json")
        path = candidate if os.path.isfile(candidate) else None
    else:
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(model_path, "generation_config.json")
            path = cached if isinstance(cached, str) else None
        except Exception:  # noqa: BLE001 — a missing hub cache only means the fallback ids are used
            path = None
    if path is None:
        return None
    with open(path) as f:
        eos = json.load(f).get("eos_token_id", [])
    return list(eos) if isinstance(eos, list) else [eos]


def _identity_page_table(num_blocks: int) -> torch.Tensor:
    """Batch-1 identity logical->physical page table, shared by prefill and decode."""
    return torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)


def _strip_template_bos(text: str, tokenizer) -> tuple[str, list[int]]:
    """The chat template emits ``<bos>`` itself and the tokenizer is expected not to add another
    (``add_bos_token=False``). When it does, drop the template's copy so the prompt carries a single BOS.
    Returns the text to encode and the ids it encodes to.

    The probe is ``encode(add_special_tokens=True)``, the same call gemma4 installs as
    ``model_args.encode_prompt(instruct=False)``, so its length is what prefill preprocessing sees."""
    probe = list(tokenizer.encode(text, add_special_tokens=True))
    bos = getattr(tokenizer, "bos_token", None)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos and bos_id is not None and text.startswith(bos) and len(probe) > 1 and probe[0] == probe[1] == bos_id:
        return text[len(bos) :], probe[1:]
    return text, probe


def _deallocate_device_tensors(roots: list, tensor_type: type) -> int:
    """Force-deallocate every ``tensor_type`` reachable from ``roots`` through model objects and containers.

    The gemma4 model has no deallocate hook and its weights sit in reference cycles, so dropping the
    generator alone leaves DRAM held until a GC pass. Only objects from ``models.*`` modules are descended
    into, so the walk stays inside the model graph (no mesh device, torch, or interpreter state). Returns
    the number deallocated."""
    seen: set[int] = set()
    stack = list(roots)
    freed = 0
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, tensor_type):
            try:
                if not hasattr(obj, "is_allocated") or obj.is_allocated():
                    obj.deallocate(True)
                    freed += 1
            except Exception:  # noqa: BLE001 — a tensor already freed by the model is not an error here
                pass
        elif isinstance(obj, dict):
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
        elif type(obj).__module__.startswith("models.") and hasattr(obj, "__dict__"):
            stack.extend(vars(obj).values())
    return freed


class DevicePromptEnhancer(PromptEnhancer):
    """On-mesh rewriter built on the ``models/demos/gemma4`` Gemma-4 E2B generator.

    Runs on the pipeline's own mesh handle (TP over axis 1, replicated over axis 0), never on the
    fixture parent or a sibling submesh: every MeshDevice handle owns an independent allocator, so two
    handles over the same chips hand out the same DRAM addresses and corrupt each other silently,
    including through trace replay. ``mesh_device`` may be None at construction; ``LTXPipeline.__init__``
    binds its submesh then.

    Built for the Blackhole Galaxy (4x8), where the DiT, the Gemma-3 encoder and this rewriter are all
    resident for the pipeline's lifetime: it loads once (``warmup``) before any trace is captured and is
    never paged out between requests. It takes no part in the dynamic_load page-in/out scheme and the
    pipeline refuses that pairing. Text-only: an I2V conditioning image is not shown to the model.
    Decode runs eagerly unless ``enable_trace``; a traced decode needs a trace region on the handle and
    must be captured before the encoder's trace."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        mesh_device=None,
        ccl_topology=None,
        max_new_tokens: int = ENHANCER_MAX_NEW_TOKENS,
        temperature: float = ENHANCER_TEMPERATURE,
        max_seq_len: int = ENHANCER_MAX_SEQ_LEN,
        top_k: int = ENHANCER_TOP_K,
        top_p: float = ENHANCER_TOP_P,
        enable_trace: bool = False,
        cache_dir: str | None = None,
    ):
        self.model_path = model_path or default_ltx_enhancer()
        self.mesh_device = mesh_device
        self.ccl_topology = ccl_topology
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.max_seq_len = int(max_seq_len)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.enable_trace = bool(enable_trace)
        assert (
            0 < self.max_new_tokens < self.max_seq_len
        ), f"max_new_tokens={self.max_new_tokens} must leave room for the prompt in max_seq_len={self.max_seq_len}"
        self.cache_dir = resolve_enhancer_cache_dir(cache_dir)
        self.last_stats: dict | None = None
        self._model_dir: str | None = None
        self._generator = None
        self._kv_cache = None
        self._tokenizer = None
        self._page_table: torch.Tensor | None = None
        self._stop_tokens: list[int] = []
        self._warm = False
        self._warned_image = False

    @property
    def name(self) -> str:
        return f"device:{self.model_path}"

    @property
    def num_page_blocks(self) -> int:
        return math.ceil(self.max_seq_len / ENHANCER_PAGE_BLOCK_SIZE)

    def is_loaded(self) -> bool:
        return self._generator is not None

    def ensure_loaded(self) -> None:
        if self._generator is not None:
            return
        if self.mesh_device is None:
            raise RuntimeError(
                "device prompt enhancer has no mesh handle: construct it through an LTXPipeline or pass mesh_device"
            )
        self._model_dir = resolve_model_dir(self.model_path)
        self._check_cache_dir_writable()
        t0 = time.time()
        with contextlib.ExitStack() as env:
            # gemma4 reads both variables while the model is built and nowhere later, so they are set for
            # that window only. TT_CACHE_PATH is the cache root as-is (no per-model namespace), so a value
            # another model exported for this process would alias its converted weights with this one's;
            # the rewriter's own model-specific dir always wins. GEMMA4_CCL_TOPOLOGY is gemma4's own
            # knob, so an explicit setting is honoured over the pipeline's fabric with a warning.
            preset_cache = os.environ.get("TT_CACHE_PATH")
            if preset_cache is not None and os.path.abspath(preset_cache) != os.path.abspath(self.cache_dir):
                logger.warning(
                    f"prompt enhancer: TT_CACHE_PATH={preset_cache!r} is set for this process but the rewriter "
                    f"caches under {self.cache_dir!r} (LTX_ENHANCER_CACHE_DIR selects it)"
                )
            env.enter_context(_scoped_env("TT_CACHE_PATH", self.cache_dir))
            if self.ccl_topology is not None:
                wanted = topology_env_value(self.ccl_topology)
                preset_topology = os.environ.get("GEMMA4_CCL_TOPOLOGY")
                if preset_topology is None:
                    env.enter_context(_scoped_env("GEMMA4_CCL_TOPOLOGY", wanted))
                elif preset_topology.strip().lower() != wanted:
                    logger.warning(
                        f"prompt enhancer: GEMMA4_CCL_TOPOLOGY={preset_topology!r} disagrees with the pipeline's "
                        f"{wanted} fabric; keeping the explicit setting"
                    )
            self._generator, self._kv_cache, self._tokenizer = self._build_generator()
        self._page_table = _identity_page_table(self.num_page_blocks)
        self._stop_tokens = resolve_stop_tokens(self._model_dir, self._tokenizer)
        self._tokenizer.stop_tokens = list(self._stop_tokens)
        logger.info(
            f"prompt enhancer loaded on device: {self._model_dir} ({time.time() - t0:.1f}s, "
            f"max_seq_len={self.max_seq_len}, stop_tokens={self._stop_tokens}, cache={self.cache_dir})"
        )

    def _check_cache_dir_writable(self) -> None:
        """gemma4 silently mirrors a cache it cannot write under ``$TT_METAL_HOME/generated``; say so up front."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            writable = os.access(self.cache_dir, os.W_OK)
        except OSError:
            writable = False
        if not writable:
            logger.warning(
                f"prompt enhancer: cache dir {self.cache_dir!r} is not writable; gemma4 will fall back to a "
                "mirror under $TT_METAL_HOME/generated (set LTX_ENHANCER_CACHE_DIR to a writable dir)"
            )

    def _build_generator(self):
        """Seam: returns ``(generator, kv_cache, tokenizer)`` for ``mesh_device``; tests substitute fakes.
        gemma4 is imported here so a pipeline without a device enhancer never pays for it."""
        from models.demos.gemma4.tt.generator import Gemma4Generator
        from models.demos.gemma4.tt.generator_trace import resolve_gemma4_bounded_sliding
        from models.tt_transformers.tt.common import PagedAttentionConfig

        # The identity page table below is the plain (unbounded) layout; bounded sliding needs per-layer
        # hybrid tables instead, so the gemma4 policy must agree with the layout this class installs.
        if resolve_gemma4_bounded_sliding(self.max_seq_len, self.mesh_device, self._model_dir):
            raise RuntimeError(
                f"gemma4 selects bounded sliding KV for max_seq_len={self.max_seq_len} on this mesh; the prompt "
                "enhancer installs only the unbounded page table (lower max_seq_len or set GEMMA4_BOUNDED_SLIDING=0)"
            )
        config = PagedAttentionConfig(block_size=ENHANCER_PAGE_BLOCK_SIZE, max_num_blocks=self.num_page_blocks)
        generator, kv_cache, tokenizer = Gemma4Generator.from_pretrained(
            mesh_device=self.mesh_device,
            model_path=self._model_dir,
            max_batch_size=1,
            max_seq_len=self.max_seq_len,
            paged_attention_config=config,
            bounded_sliding_kv_cache=False,
        )
        return generator, kv_cache, tokenizer

    def warmup(self) -> None:
        """Load and run one short greedy rewrite per prefill bucket (``WARMUP_PROMPTS``) so prefill and decode
        kernels compile before the first request. The generator's own prefill sweep is skipped: every
        prefill here runs untraced. Compiled programs live on the mesh handle, so this runs once per
        process."""
        self.ensure_loaded()
        if self._warm:
            return
        from models.tt_transformers.tt.common import get_padded_prefill_len

        t0 = time.time()
        warmed = set()
        for prompt in WARMUP_PROMPTS:
            self._generate(build_messages(prompt, "t2v"), seed=0, max_new_tokens=WARMUP_NEW_TOKENS, temperature=0.0)
            bucket = get_padded_prefill_len(self.last_stats["prompt_tokens"])
            warmed.add(bucket)
            logger.info(f"prompt enhancer warm prompt: {self.last_stats['prompt_tokens']} tokens -> {bucket} bucket")
        # The longest prompt that fits lands in this bucket, as does every stock LTX prompt.
        production_bucket = get_padded_prefill_len(self.max_seq_len - self.max_new_tokens)
        if production_bucket not in warmed:
            logger.warning(
                f"prompt enhancer warmup compiled prefill buckets {sorted(warmed)} but not {production_bucket}; "
                "the first full-length request pays that compile"
            )
        self._warm = True
        logger.info(f"prompt enhancer warm ({time.time() - t0:.1f}s)")

    def unload(self) -> None:
        """Drop the generator and force-free the device tensors it owns. Idempotent."""
        generator, self._generator = self._generator, None
        kv_cache, self._kv_cache = self._kv_cache, None
        self._tokenizer = None
        self._page_table = None
        if generator is None and kv_cache is None:
            return
        if generator is not None and hasattr(generator, "release_persistent_capture"):
            try:
                generator.release_persistent_capture()
            except Exception as e:  # noqa: BLE001 — traces are best-effort cleanup; the weights still go
                logger.warning(f"prompt enhancer: releasing traces failed: {e!r}")
        freed = _deallocate_device_tensors([generator, kv_cache], self._device_tensor_type())
        del generator, kv_cache
        gc.collect()
        logger.info(f"prompt enhancer unloaded ({freed} device tensors freed)")

    @staticmethod
    def _device_tensor_type() -> type:
        import ttnn

        return ttnn.Tensor

    def enhance(
        self,
        prompt: str,
        *,
        mode: EnhanceMode = "t2v",
        image_path: str | None = None,
        seed: int = ENHANCER_SEED,
    ) -> str:
        self.ensure_loaded()
        if image_path is not None and not self._warned_image:
            logger.warning(
                "device prompt enhancer is text-only; the I2V conditioning image is not shown to the rewriter"
            )
            self._warned_image = True
        return self._generate(
            build_messages(prompt, mode), seed=seed, max_new_tokens=self.max_new_tokens, temperature=self.temperature
        )

    def _generate(self, messages: list[dict[str, str]], *, seed: int, max_new_tokens: int, temperature: float) -> str:
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = self._tokenize_prompt(text, max_new_tokens)
        prompt_len = len(prompt_ids)
        assert (
            prompt_len + max_new_tokens <= self.max_seq_len
        ), f"prompt of {prompt_len} tokens + {max_new_tokens} new tokens exceeds max_seq_len={self.max_seq_len}"
        if temperature > 0:
            torch.manual_seed(seed)

        t0 = time.time()
        logits = self._generator.prefill_forward_text(
            torch.tensor(prompt_ids, dtype=torch.int32).view(1, prompt_len),
            page_table=self._page_table,
            kv_cache=self._kv_cache,
            prompt_lens=[prompt_len],
            warmup_prefill=False,
            enable_trace=False,
            sampling_params=None,
        )
        tok = self._sample(logits, temperature)
        prefill_s = time.time() - t0

        # The prefill's token is the first new token; each decode step feeds the previous token back and
        # advances the position by one. Positions are not advanced by the generator itself.
        new_ids: list[int] = []
        current_pos = torch.tensor([prompt_len])
        decode_steps = 0
        t0 = time.time()
        while tok not in self._stop_tokens and len(new_ids) < max_new_tokens:
            new_ids.append(tok)
            if len(new_ids) >= max_new_tokens:
                break
            # reload_inputs=True: E2B's per-layer inputs are computed on the host for every token.
            logits, _ = self._generator.decode_forward(
                torch.tensor([[tok]]),
                current_pos,
                enable_trace=self.enable_trace,
                page_table=self._page_table,
                kv_cache=self._kv_cache,
                sampling_params=None,
                reload_inputs=True,
                reload_page_table=False,
                reload_sampling_params=False,
                reset_sampling_state=False,
            )
            current_pos += 1
            decode_steps += 1
            tok = self._sample(logits, temperature)
        decode_s = time.time() - t0

        self.last_stats = {
            "prompt_tokens": prompt_len,
            "new_tokens": len(new_ids),
            "prefill_s": round(prefill_s, 3),
            "decode_s": round(decode_s, 3),
            "tok_s": round(decode_steps / decode_s, 3) if decode_steps and decode_s > 0 else None,
        }
        return self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def _tokenize_prompt(self, text: str, max_new_tokens: int) -> list[int]:
        """Seam: templated text -> prompt ids through the generator's own prefill preprocessing (instruct=False:
        the text is already templated). A prompt that would be left-clipped to fit is rejected instead, since
        clipping removes the system prompt's head."""
        from models.tt_transformers.tt.common import preprocess_inputs_prefill

        text, probe = _strip_template_bos(text, self._tokenizer)
        _, encoded, _, _ = preprocess_inputs_prefill(
            [text], self._tokenizer, self._generator.model_args, False, max_new_tokens, max_prefill_len=self.max_seq_len
        )
        prompt_ids = [int(t) for t in encoded[0]]
        if len(prompt_ids) < len(probe):
            raise PromptTooLongError(
                f"prompt of {len(probe)} tokens does not fit max_seq_len={self.max_seq_len} with "
                f"{max_new_tokens} new tokens"
            )
        return prompt_ids

    def _sample(self, logits: torch.Tensor, temperature: float) -> int:
        """Host sampling over the last position's logits: argmax at temperature 0, else temperature with
        top-k then top-p, the checkpoint's own nucleus settings. ``sample_host`` has no top-k and its
        default top_p is not the model's, so the top-k cut is applied here and top_p always passed."""
        last = logits.reshape(-1, logits.shape[-1])[-1:]
        if temperature <= 0:
            return int(last.argmax().item())
        from models.tt_transformers.tt.common import sample_host

        if 0 < self.top_k < last.shape[-1]:
            kth = torch.topk(last, self.top_k, dim=-1).values[..., -1:]
            last = last.masked_fill(last < kth, float("-inf"))
        _, tok = sample_host(last, temperature=temperature, top_p=self.top_p)
        return int(tok.reshape(-1)[0].item())


def build_prompt_enhancer(
    backend: str | None,
    *,
    model_path: str | None = None,
    mesh_device=None,
) -> PromptEnhancer | None:
    """Map the ``LTX_PROMPT_ENHANCER`` setting to an enhancer. Unset or ``off`` keeps the raw prompt.

    ``mesh_device`` is normally left None: the pipeline the enhancer is handed to binds its own handle."""
    key = (backend or "").strip().lower()
    if key in ("", "0", "off", "none", "false"):
        return None
    if key == "host":
        return HostPromptEnhancer(model_path)
    if key == "device":
        return DevicePromptEnhancer(model_path, mesh_device=mesh_device)
    raise ValueError(f"LTX_PROMPT_ENHANCER={backend!r}: expected 'host', 'device', or unset")


def apply_prompt_enhancer(
    enhancer: PromptEnhancer | None,
    prompt: str,
    *,
    image_path: str | None = None,
    mode: EnhanceMode | None = None,
    seed: int = ENHANCER_SEED,
) -> str:
    """Rewrite ``prompt`` through ``enhancer``; identity when there is none.

    ``mode`` follows the conditioning image unless given explicitly, so a caller cannot pass an I2V
    frame and silently get the T2V instructions. The raw prompt goes through unchanged when the rewrite
    comes back empty (the reference instructs the model to return the original on invalid input) or when
    the prompt does not fit the rewriter's context (``PromptTooLongError``): a prompt the encoder can take
    must never fail a request because the rewriter cannot."""
    if enhancer is None:
        return prompt
    if mode is None:
        mode = "i2v" if image_path else "t2v"
    enhancer.ensure_loaded()
    t0 = time.time()
    try:
        enhanced = enhancer.enhance(prompt, mode=mode, image_path=image_path, seed=seed).strip()
    except PromptTooLongError as e:
        logger.warning(f"prompt enhancer {enhancer.name} skipped, keeping the raw prompt: {e}")
        return prompt
    if not enhanced:
        logger.warning(f"prompt enhancer {enhancer.name} returned nothing; keeping the raw prompt")
        return prompt
    logger.info(
        f"prompt enhanced ({enhancer.name}, {mode}, {time.time() - t0:.1f}s)\n  raw: {prompt!r}\n  enhanced: {enhanced!r}"
    )
    return enhanced
