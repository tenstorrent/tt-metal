"""What a model retires per call is read from the model, not looked up by its task.

The unit of work -- token, denoise step, or one forward pass -- decides the whole roofline: which
rate the ceiling is in, what the band means, and what a reader quotes. It was resolved from a fixed
table keyed on the HF `pipeline_tag`, with the class name in config.json as fallback.

That cannot be right, because the unit is a property of the ARCHITECTURE and neither input states it.
`text-to-speech` covers XTTS and VibeVoice, which emit tokens one at a time, and Kokoro-82M, which is
StyleTTS2: text in, whole waveform out, no loop anywhere. One tag, two units. The table has to pick,
so it picks the common case and is wrong for the rest:

    Kokoro-82M          tagged text-to-speech  -> token   should be inference
    HunyuanImage-3.0    tagged text-to-image   -> step    should be token (HunyuanImage3ForCausalMM
                                                          is autoregressive, not a denoise loop)

No value HuggingFace could publish fixes this: a tag names the task, and the same task is served by
different machines. Nor does keeping the table current -- it is 49 hand-written entries against a
taxonomy HF keeps extending, and the code already notes it is missing 19 of them.

So the answer is not a better table. It is to stop asking a question the input cannot answer, and use
what the built pipeline actually does.

TWO THINGS ARE FIXED HERE.

The observed unit now WINS over the config guess. Optimize resolves the unit at the start of a run
from config.json while the pipeline it is about to profile sits right there -- there is no
cross-tool constraint forcing that, it is simply what the ceiling code always did. The guess survives
only for the window before the first trace, where nothing else exists.

And the observation is now STRUCTURAL. It was

    decode = next((r for r in results if "decode" in r[0].lower()), None)

-- a substring test on free text emit-e2e wrote, which is a guess wearing an observation's clothes: a
pipeline whose recurring stage is called `generate` reads as one-pass, and one that names any stage
`decode` reads as autoregressive whether it loops or not. The decode CONTRACT is the real signal. A
pipeline exposing decode_step(state) retires one token per call by definition -- it is what
PipelineDecodeAdapter requires and raises NotTraceCapable without.
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RUN = Path(__file__).resolve().parent.parent / "cc_optimize" / "run.py"
TR = Path(__file__).resolve().parent.parent / "agent" / "trace_replay.py"


# ---------------------------------------------------------------- the ceiling has ONE source


def test_the_ceiling_unit_comes_only_from_the_observation():
    """THE CHANGE. A wrong unit does not degrade the ceiling, it puts it in the wrong CURRENCY -- and
    the band, the at-floor verdict and the headline rate all inherit that. So the tag table no longer
    feeds it at all."""
    src = RUN.read_text()
    i = src.index('facts["unit"] = _observed')  # the CEILING site, not the byte-walk one
    assert "PERF_MCP_LAST_HEADLINE_UNIT" in src[max(0, i - 400) : i]
    assert "unit_from_config" not in src, "the tag guess still backs the ceiling unit"


def test_no_observation_means_no_unit_rather_than_a_guessed_one():
    """The rule the rest of the code already follows -- _anchored_ceiling_facts: "No recoverable unit
    means no ceiling, which lands on the floor fallback: weaker, but not wrong." """
    src = RUN.read_text()
    i = src.index('facts["unit"] = _observed')
    assert "else:" not in src[i : i + 200], "an else branch would reintroduce the guess"


def test_the_tag_still_chooses_the_MEASUREMENT_CONDITIONS():
    """ISL/OSL/steps/resolution must be picked BEFORE anything runs, so no observation can supply
    them -- and a wrong guess there shows up in the scorecard rather than rescaling the ceiling."""
    src = RUN.read_text()
    assert "unit_for_tag" in src, "default_conditions lost its unit"
    i = src.index("unit_for_tag")
    assert "pipeline_tag" in src[i : i + 200]


# ---------------------------------------------------------------- the observation is structural


# ---------------------------------------------------------------- the decision, as a pure function


def _hu(names, pipe=None):
    from agent.perf_adapter import headline_unit

    return headline_unit(names, pipe)


class _WithContract:
    def decode_step(self, state):
        return state


def test_the_decode_contract_marks_a_token_unit_whatever_the_stage_is_called():
    """The structural claim: decode_step(state) retires one token per call BY DEFINITION, so the
    stage may be called anything at all."""
    for name in ("generate", "autoregressive", "decode", "loop", ""):
        assert _hu([name], _WithContract()) == "token", name


def test_no_contract_and_no_matching_name_is_one_pass():
    """Kokoro's shape: stages, no decode_step, nothing named decode -> a single forward."""
    assert _hu(["synthesize"], object()) == "inference"
    assert _hu(["encode", "vocode"], None) == "inference"


def test_a_step_unit_is_declared_not_spelled():
    """A stage CALLED denoise is told apart from one that IS a denoise step by what the model says.

    This used to read the word: "denoise" or "diffus" anywhere in a stage name returned "step". That
    is a guess about vocabulary -- it fires on a stage merely named that way and stays silent for a
    diffusion model whose stage is called anything else. PIPELINE_UNIT is the model stating it.
    """
    assert _hu(["denoise"], None) == "inference"
    assert _hu(["diffusion_step"], None) == "inference"

    class _Declared:
        PIPELINE_UNIT = "step"

    assert _hu(["anything_at_all"], _Declared()) == "step"


def test_a_stage_adapter_pipeline_is_served_by_its_item_count():
    """It exposes per-stage hooks rather than the single decode contract, so it is asked what one
    call retires -- <stage>_trace_items(), the same seam _Stage derives `recurring` from. The name
    is not consulted: a stage called `decode` that states nothing says nothing."""
    assert _hu(["prefill", "decode"], None) == "inference"

    class _Counts:
        def decode_trace_items(self):
            return 1

        def prefill_trace_items(self):
            return 128

    assert _hu(["prefill", "decode"], _Counts()) == "token"

    class _NoneRecur:
        def encode_trace_items(self):
            return 1500

    assert _hu(["encode"], _NoneRecur()) == "inference"


def test_the_contract_outranks_a_conflicting_name():
    """A pipeline that HAS decode_step but names its stage 'denoise' is autoregressive, not diffusion."""
    assert _hu(["denoise"], _WithContract()) == "token"


def test_it_never_raises_on_junk():
    """It runs on every trace; a bad stage list must degrade to one-pass, not end the measurement."""
    for bad in (None, [], [None], [123]):
        assert _hu(bad, None) in ("token", "step", "inference")


# ---------------------------------------------------------------- the override reaches a CACHED file


def test_the_cached_facts_file_does_not_freeze_the_guess():
    """perf_target_inputs.json is written ONCE at setup -- before any trace exists -- and read from
    cache for the rest of the run. Correcting only the producer would never reach a run whose file
    already exists, so the override is applied on the LOAD path, which both routes go through."""
    src = (Path(__file__).resolve().parent.parent / "cc_optimize" / "perf_mcp.py").read_text()
    i = src.index("def _load_perf_target_inputs")
    body = src[i : src.index("\ndef ", i + 1)]
    assert "PERF_MCP_LAST_HEADLINE_UNIT" in body, "the cached path still returns the stale unit"
    assert body.count("return facts") == 1, "both routes must fall through the same override"


def test_the_override_only_fires_when_something_was_observed():
    """Before the first trace there is no observation, and the guess must stand -- otherwise round one
    has no unit and no ceiling at all."""
    src = (Path(__file__).resolve().parent.parent / "cc_optimize" / "perf_mcp.py").read_text()
    i = src.index('_obs = str(os.environ.get("PERF_MCP_LAST_HEADLINE_UNIT")')
    assert "if _obs and" in src[i : i + 300]


# ---------------------------------------------------------------- resolution reaches the perf side


def test_resolution_is_recorded_in_the_facts():
    """emit-e2e reads image_size to build its PCC input (e2e_emitter: vision_config.image_size ->
    torch.randn(1, 3, H, W)); the perf side had NO notion of it, so a steps/s or vision inferences/s
    figure could not state the resolution it described -- and resolution IS the work, a denoise step
    at 1024 being ~4x the step at 512."""
    src = RUN.read_text()
    assert "resolution_from_config" in src
    i = src.index("resolution_from_config")
    assert 'facts["resolution"]' in src[i : i + 400]


def test_an_operator_can_override_the_resolution():
    """So it can be swept, the way TT_PERF_BATCH sweeps batch."""
    assert "TT_PERF_RESOLUTION" in RUN.read_text()


def test_the_scorecard_prints_it_only_when_the_model_has_one():
    """A resolution line on a text model would state a condition that never existed."""
    src = RUN.read_text()
    i = src.index("resolution        :")
    assert "if _res:" in src[max(0, i - 300) : i]


def test_the_byte_walk_exclusion_uses_the_observation_too():
    """The same defect one layer down. The lookup-only exclusion drops the embedding table from the
    streamed bytes because a token unit reads it by INDEX, one row per token -- but whether the model
    IS a token unit came from the tag. Kokoro-82M is tagged text-to-speech, reads as `token`, and has
    no token loop, so its tables would be excluded from a byte count they belong in -- and the ceiling
    is peak_BW over exactly those bytes."""
    src = RUN.read_text()
    # Anchor on the ASSIGNMENT that reads the observation, not on the first `_unit = ` in the file --
    # a definite-assignment guard (`_unit = ""`) now precedes it and would capture a bare prefix.
    i = src.index("_unit = str(os.environ.get(")
    assert "PERF_MCP_LAST_HEADLINE_UNIT" in src[i : i + 300], src[i : i + 300]
    assert "unit_for_tag" in src[i : i + 400], "the pre-trace fallback was dropped"
