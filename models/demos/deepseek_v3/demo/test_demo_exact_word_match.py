import difflib
import json
import re
import subprocess
from pathlib import Path

import pytest

# === Paths ====================================================================
MODEL_PATH = Path("models/demos/deepseek_v3/reference")
DEMO_SCRIPT = Path("models/demos/deepseek_v3/demo/demo.py")
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


# === Demo stdout parsing ======================================================
_PROMPT_HDR_RE = re.compile(r"^Prompt\[(\d+)\]:\s*(.*)$")
_GEN_HDR_RE = re.compile(r"^Generation\[(\d+)\]:")


def extract_all_generations(stdout: str):
    """
    Demo prints:
        ===== Generated =====

        ------------------------------
        Prompt[1]: <prompt text>
        Generation[1]:
        <gen text...>
        ------------------------------
        Prompt[2]: ...
        Generation[2]:
        <gen text...>
        ------------------------------
        =====================

    Return a list of tuples: [(prompt_text, gen_text), ...] preserving order.
    """
    # Focus only on the block between the markers to reduce noise
    start = "===== Generated ====="
    end = "====================="
    if start in stdout and end in stdout:
        block = stdout.split(start, 1)[1].split(end, 1)[0]
    else:
        block = stdout

    lines = block.splitlines()

    results = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m_prompt = _PROMPT_HDR_RE.match(line)
        if m_prompt:
            idx = int(m_prompt.group(1))
            prompt_text = m_prompt.group(2)
            # Next non-empty, expect Generation[idx]:
            i += 1
            while i < len(lines) and not _GEN_HDR_RE.match(lines[i].strip()):
                i += 1
            if i >= len(lines):
                # Malformed: no Generation header
                results.append((prompt_text, ""))
                break
            # Consume "Generation[idx]:" line
            i += 1
            # Accumulate generation text until a dashed separator or next Prompt[...] or end
            gen_lines = []
            while i < len(lines):
                s = lines[i]
                if s.strip().startswith("-" * 10):
                    break
                if _PROMPT_HDR_RE.match(s.strip()):
                    # We've hit the next prompt block without a dashed line
                    i -= 1  # step back so outer loop sees this as a prompt header
                    break
                gen_lines.append(s)
                i += 1
            gen_text = "\n".join(gen_lines).strip()
            results.append((prompt_text, gen_text))
        i += 1

    return results


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


def test_multi_prompt_generation_matches_reference(tmp_path):
    """
    Loads a JSON dict {prompt: expected_text}, executes the demo ONCE with up to 32 prompts,
    parses all generations, and validates each against its expected reference.

    Defaults to case/punctuation-sensitive exact match (WER==0); see flags above.
    """
    cache_dir = tmp_path / "cache"
    ref_map = load_reference_map(REFERENCE_JSON)

    # Respect demo's 32-prompt limit (stable order using JSON key order if Python 3.7+)
    all_prompts = list(ref_map.keys())
    prompts = all_prompts[:MAX_PROMPTS]

    # Sanity: ensure we have at least one prompt
    assert len(prompts) > 0, "No prompts found in reference JSON."

    # Run the demo once with ALL prompts
    cmd = [
        "python",
        str(DEMO_SCRIPT),
        *prompts,
        "--model-path",
        str(MODEL_PATH),
        "--cache-dir",
        str(cache_dir),
        "--max-new-tokens",
        "200",
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,  # capture stdout
        stderr=subprocess.STDOUT,  # fold stderr into stdout (helps debugging)
        text=True,  # decode to str
        check=False,
    )

    assert proc.returncode == 0, (
        "demo.py failed.\n" f"Exit code: {proc.returncode}\n" f"--- stdout/stderr (tail) ---\n{proc.stdout[-4000:]}"
    )

    parsed = extract_all_generations(proc.stdout)
    assert len(parsed) == len(prompts), (
        f"Parsed {len(parsed)} generations but passed {len(prompts)} prompts.\n"
        f"--- stdout/stderr (tail) ---\n{proc.stdout[-4000:]}"
    )

    # Build prompt -> generated map (the demo prints each block with its prompt)
    gen_map: dict[str, str] = {}
    for p_text, g_text in parsed:
        gen_map[p_text] = g_text

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
        message = (
            f"{len(failures)} prompt(s) failed the word-accuracy check.\n\n"
            + "\n\n".join(failures)
            + "\n\n--- Raw tail of stdout ---\n"
            + proc.stdout[-2000:]
        )
        pytest.fail(message)

    # If we get here, all prompts passed
    assert True
