import difflib
import json
import re
from pathlib import Path

import pytest

from models.demos.deepseek_v3.demo.demo import run_demo

# === Paths ====================================================================
MODEL_PATH = Path("/proj_sw/user_dev/deepseek-ai/DeepSeek-R1-0528")  # Path("models/demos/deepseek_v3/reference")
REFERENCE_JSON = Path("models/demos/deepseek_v3/demo/deepseek_32_prompts_outputs.json")

# === Comparison options =======================================================
IGNORE_CASE = False
STRIP_PUNCT = False

# By default require exact match (WER == 0)
REQUIRE_EXACT_MATCH = True
WER_TOLERANCE = 0.0

# Limit of prompts to pass in one run (demo supports up to 32)
MAX_PROMPTS = 32

# === Tokenization / diff utilities ===========================================
_WORD_RE = re.compile(r"\w+('\w+)?|\S", flags=re.UNICODE)


def _normalize_text(s: str, ignore_case: bool, strip_punct: bool):
    if ignore_case:
        s = s.lower()
    toks = _WORD_RE.findall(s)
    if strip_punct:
        toks = [t for t in toks if re.search(r"\w", t)]
    return toks


def word_diff_metrics(generated: str, reference: str, *, ignore_case=True, strip_punct=True):
    """Return dict with WER, exact_match, matched/total counts, and first diffs."""
    gen_tokens = _normalize_text(generated, ignore_case, strip_punct)
    ref_tokens = _normalize_text(reference, ignore_case, strip_punct)

    sm = difflib.SequenceMatcher(a=ref_tokens, b=gen_tokens, autojunk=False)
    opcodes = sm.get_opcodes()

    subs = sum(1 for tag, *_ in opcodes if tag == "replace")
    dels = sum((i2 - i1) for tag, i1, i2, *_ in opcodes if tag == "delete")
    ins = sum((j2 - j1) for tag, *_, j1, j2 in opcodes if tag == "insert")
    N = max(1, len(ref_tokens))
    wer = (subs + dels + ins) / N

    matched = len(ref_tokens) - subs - dels
    exact = (wer == 0.0) and (len(gen_tokens) == len(ref_tokens))

    diffs = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        diffs.append(
            {
                "tag": tag,
                "ref": " ".join(ref_tokens[i1:i2]),
                "gen": " ".join(gen_tokens[j1:j2]),
                "ctx_left": " ".join(ref_tokens[max(0, i1 - 3) : i1]),
                "ctx_right": " ".join(ref_tokens[i2 : i2 + 3]),
            }
        )

    return {
        "wer": wer,
        "exact": exact,
        "matched": matched,
        "total": len(ref_tokens),
        "gen_count": len(gen_tokens),
        "ref_count": len(ref_tokens),
        "diffs": diffs,
    }


# === Reference JSON loading ===================================================
def load_reference_map(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Reference JSON must be a JSON object mapping prompts to expected texts.")
    ref_map: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(v, str):
            ref_map[k] = v
        elif isinstance(v, dict) and "text" in v and isinstance(v["text"], str):
            ref_map[k] = v["text"]
        else:
            raise ValueError(f"Invalid reference for prompt {k!r}: expected string or object with 'text' field.")
    if not ref_map:
        raise ValueError("Reference JSON is empty; nothing to test.")
    return ref_map


@pytest.mark.parametrize(
    "max_new_tokens",
    [200],
)
def test_multi_prompt_generation_matches_reference(tmp_path, max_new_tokens):
    """
    Loads a JSON dict {prompt: expected_text}, executes the demo ONCE with up to 32 prompts
    by calling run_demo(), and validates each returned generation['text'] against its reference.

    Defaults to case/punctuation-sensitive exact match (WER==0); see flags above.
    """
    # cache_dir = tmp_path / "cache"
    cache_dir = Path("/proj_sw/user_dev/deepseek-v3-cache")
    ref_map = load_reference_map(REFERENCE_JSON)

    # Respect demo's 32-prompt limit
    all_prompts = list(ref_map.keys())
    prompts = all_prompts[:MAX_PROMPTS]

    # Ensure at least one prompt
    assert len(prompts) > 0, "No prompts found in reference JSON."

    # Run the demo once with ALL prompts
    results = run_demo(
        prompts=prompts,
        model_path=str(MODEL_PATH),
        max_new_tokens=max_new_tokens,
        cache_dir=str(cache_dir),
        random_weights=False,
        token_accuracy=False,
        early_print_first_user=False,
    )

    assert isinstance(results, dict) and "generations" in results, "run_demo() did not return expected structure."
    generations = results["generations"]
    assert isinstance(generations, list), "run_demo()['generations'] must be a list."
    assert len(generations) == len(prompts), f"Got {len(generations)} generations but passed {len(prompts)} prompts."

    # Build prompt -> generated map by index (run_demo preserves order)
    gen_map: dict[str, str] = {}
    for i, gen_entry in enumerate(generations):
        prompt = prompts[i]
        text = gen_entry.get("text")
        # In random-weights mode text may be None; we are not in that mode here.
        assert isinstance(text, str), f"Generation for prompt index {i} has no 'text'."
        gen_map[prompt] = text

    # Verify every prompt we passed has a corresponding generation
    missing = [p for p in prompts if p not in gen_map]
    assert not missing, "Missing generations for prompts:\n" + "\n".join(f"- {m!r}" for m in missing)

    # Compare each generation to reference; collect failures to report together
    failures = []
    for p in prompts:
        generated_text = gen_map[p]
        reference_text = ref_map[p]

        metrics = word_diff_metrics(
            generated_text,
            reference_text,
            ignore_case=IGNORE_CASE,
            strip_punct=STRIP_PUNCT,
        )

        if REQUIRE_EXACT_MATCH:
            ok = metrics["exact"]
            threshold_msg = "exact match required (WER==0)"
        else:
            ok = metrics["wer"] <= float(WER_TOLERANCE)
            threshold_msg = f"WER <= {WER_TOLERANCE:.3f}"

        if not ok:
            # Prepare a concise diff section for this prompt
            diff_lines = []
            for i, d in enumerate(metrics["diffs"][:20], 1):
                diff_lines.append(
                    f"    [{i}] {d['tag'].upper()}\n"
                    f"      ref: {d['ref']}\n"
                    f"      gen: {d['gen']}\n"
                    f"      ctx: ... {d['ctx_left']} | {d['ctx_right']} ..."
                )
            failures.append(
                f"Prompt: {p!r}\n"
                f"  {threshold_msg}\n"
                f"  WER={metrics['wer']:.4f}  matched={metrics['matched']}/{metrics['total']} "
                f"(gen={metrics['gen_count']}, ref={metrics['ref_count']})\n"
                + ("\n".join(diff_lines) if diff_lines else "  (no diff ops captured)")
            )

    if failures:
        message = f"{len(failures)} prompt(s) failed the word-accuracy check.\n\n" + "\n\n".join(failures)
        pytest.fail(message)

    # If we get here, all prompts passed
    assert True
