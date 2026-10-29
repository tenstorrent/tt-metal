# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""Portable NLLB trained generation regression. No CHIA or archived-run imports."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import numpy as np

CASES = ("base", "singles", "reverse", "padding")

PARAGRAPHS = (
    "On Saturday morning, I walked to the public library to return a book about the history of our town. "
    "The librarian showed me a collection of old photographs and explained how the railway station had changed "
    "over the years. After reading for an hour, I borrowed another book and went outside to catch the bus. "
    "It was delayed by the rain, so I called my sister and told her that I would arrive at her house a little later "
    "than planned. She was preparing lunch with vegetables from her garden and invited our neighbors to join us. "
    "During the afternoon we talked about the upcoming school festival, organized the books that the children "
    "wanted to donate, and wrote a list of supplies for the community kitchen. Before sunset we walked along "
    "the river, where a group of volunteers was repairing the benches near the old bridge. They explained that "
    "the park would reopen the following week after the damaged paths had been restored. We offered to help "
    "plant some flowers and arranged to meet them again on Sunday morning. On our way home we stopped at a "
    "small bakery, bought fresh bread for dinner, and thanked the owner for supporting the local school. "
    "That evening I finished reading the first chapter of my new book and wrote a letter to a friend who "
    "had moved abroad. I described the changes in our neighborhood and asked about her new job and family.",
    "The research group met early on Monday to prepare a report about the water quality in nearby villages. "
    "Each student brought notes from a different location, together with photographs and measurements collected "
    "during the previous month. The team compared the results, checked the dates on every sample, and discussed "
    "which observations needed further investigation. They decided to visit two schools and speak with the "
    "teachers about safe drinking water. After lunch, the students packed their equipment and traveled to the "
    "first village by bus. The school principal welcomed them and introduced them to a class that had recently "
    "started a science project. The children asked questions about the river, the rain, and the farms surrounding "
    "their homes. One teacher showed the visitors a notebook containing weather records from the past ten years. "
    "The researchers explained how they would compare those records with their measurements without drawing "
    "conclusions before the analysis was complete. In the afternoon, they walked to the public well with a "
    "local engineer who described the recent repairs to the pump. They collected another sample and carefully "
    "recorded the location, temperature, and time. Before returning to the university, the group thanked the "
    "families who had helped organize the visit and promised to share the final report in clear language. "
    "The next morning they reviewed their notes once more and planned a second visit to answer the remaining questions.",
    "The train has arrived.",
    "",
)


def build_inputs(tokenizer, config):
    """Natural source prefixes, intentionally truncated to exact boundary lengths."""
    source, target = [int(tokenizer.convert_tokens_to_ids(lang)) for lang in ("eng_Latn", "fra_Latn")]
    if source == target or any(not 3 <= token < config["vocab_size"] for token in (source, target)):
        raise ValueError("invalid source/target language IDs")
    rows, original_lengths = [], []
    for index, text in enumerate(PARAGRAPHS):
        ids = list(tokenizer.encode(text, add_special_tokens=True))
        original_lengths.append(len(ids))
        if not ids or ids[0] != source or ids[-1] != config["eos_token_id"]:
            raise ValueError("tokenizer source language/EOS placement differs")
        if (
            any(type(token) is not int or not 0 <= token < config["vocab_size"] for token in ids)
            or config["pad_token_id"] in ids
            or config["eos_token_id"] in ids[1:-1]
        ):
            raise ValueError("invalid tokenizer body tokens")
        if index < 2:
            length = 255 + index
            if len(ids) < length:
                raise ValueError("natural source is too short for real-token boundary")
            ids = ids[:length]
            ids[-1] = config["eos_token_id"]
        rows.append(ids)
    lengths = [len(row) for row in rows]
    if not all(2 <= length < 255 for length in lengths[2:]):
        raise ValueError("short/empty text no longer produces short rows")
    inputs = dict(
        input_ids=np.full((4, 256), config["pad_token_id"], dtype=np.int64),
        attention_mask=np.zeros((4, 256), dtype=np.int64),
    )
    for index, row in enumerate(rows):
        inputs["input_ids"][index, : len(row)] = row
        inputs["attention_mask"][index, : len(row)] = 1
    return inputs, dict(
        source_id=source, target_id=target, source_lengths=lengths, untruncated_source_lengths=original_lengths
    )


def requests(case, inputs, lengths):
    ids, mask = inputs["input_ids"], inputs["attention_mask"]
    if case == "base":
        return [("cap63", ids, mask, 63), ("cap64", ids, mask, 64)]
    if case == "singles":
        return [(f"row{i}", ids[i : i + 1, :length], mask[i : i + 1, :length], 64) for i, length in enumerate(lengths)]
    if case == "reverse":
        return [("cap64", ids[::-1].copy(), mask[::-1].copy(), 64)]
    if case == "padding":
        return [("cap64", ids[:1], mask[:1], 64)]
    raise ValueError("unknown envelope subcase")


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(json.dumps([value.dtype.str, list(value.shape)]).encode() + value.tobytes()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(tokens, batch, target, cap, config):
    if (
        not isinstance(tokens, np.ndarray)
        or tokens.dtype != np.int64
        or tokens.ndim != 2
        or tokens.shape[0] != batch
        or not 2 <= tokens.shape[1] <= cap + 1
        or np.any(tokens < 0)
        or np.any(tokens >= config["vocab_size"])
    ):
        raise AssertionError("invalid output shape/dtype/token range")
    rows = []
    for row in tokens.tolist():
        if row[:2] != [config["decoder_start_token_id"], target]:
            raise AssertionError("decoder start or target language changed")
        eos = row.index(config["eos_token_id"], 2) if config["eos_token_id"] in row[2:] else None
        end = len(row) if eos is None else eos + 1
        if config["pad_token_id"] in row[:end] or any(x != config["pad_token_id"] for x in row[end:]):
            raise AssertionError("padding before EOS or nonpadding after EOS")
        if eos is None and end != cap + 1:
            raise AssertionError("generation stopped without EOS before requested cap")
        rows.append(row[:end])
    return rows


def observe_request(model, ids, mask, target, cap, synchronize, deadline):
    original, instance_override = model.decode, "decode" in model.__dict__
    events, caches = [], []
    supplied_ids, supplied_mask = ids.copy(), mask.copy()

    def observe(prefix, *args, **kwargs):
        if time.monotonic() > deadline:
            raise TimeoutError("generation regression deadline")
        before = prefix.copy()
        cache = kwargs.get("cross_kv")
        if kwargs.get("final_token_only") is not True or not isinstance(cache, dict):
            raise AssertionError("actual cached last-token path not exercised")
        if prefix.shape[1] == 2:
            if cache or any(cache is previous for previous in caches):
                raise AssertionError("cache reused across requests or rows")
            caches.append(cache)
        result = original(prefix, *args, **kwargs)
        if not np.array_equal(before, prefix):
            raise AssertionError("decoder prefix mutated")
        if not cache:
            raise AssertionError("cross-KV cache was never populated")
        logits = np.asarray(result)
        if logits.shape != (1, 1, model.config["vocab_size"]) or not np.isfinite(logits).all():
            raise AssertionError("invalid observed logits")
        events.append((before[0].tolist(), int(logits[0, 0].argmax())))
        return result

    model.decode = observe
    try:
        synchronize()
        tokens = np.asarray(model.generate(supplied_ids, supplied_mask, target, cap)).copy()
        synchronize()
        rows = canonical(tokens, len(ids), target, cap, model.config)
        expected = [(row[:n], row[n]) for row in rows for n in range(2, len(row))]
        if events != expected or len(caches) != len(ids) or any(caches):
            raise AssertionError("observed decode events or cache cleanup disagree")
        return tokens, dict(
            observed_prefixes=[list(range(2, len(row))) for row in rows],
            actual_new_tokens=[len(row) - 1 for row in rows],
            cache_cleared=True,
            input_preserved=True,
        )
    finally:
        if instance_override:
            model.decode = original
        else:
            delattr(model, "decode")
        if not np.array_equal(ids, supplied_ids) or not np.array_equal(mask, supplied_mask):
            raise AssertionError("source inputs mutated")


def behavior_checks(outputs, observations, target, config):
    base = canonical(outputs["base__cap64"], 4, target, 64, config)
    long_coverage = all(all(n in observations["base__cap64"]["observed_prefixes"][i] for n in (63, 64)) for i in (0, 1))
    return dict(
        actual_boundary_63_64=long_coverage,
        mixed_natural_eos=any(row[-1] == config["eos_token_id"] for row in base[2:]),
        cap_prefix=canonical(outputs["base__cap63"], 4, target, 63, config) == [row[:64] for row in base],
        batch_single=[canonical(outputs[f"singles__row{i}"], 1, target, 64, config)[0] for i in range(4)] == base,
        reverse=canonical(outputs["reverse__cap64"], 4, target, 64, config) == base[::-1],
        right_padding=canonical(outputs["padding__cap64"], 1, target, 64, config)[0]
        == canonical(outputs["singles__row0"], 1, target, 64, config)[0]
        == base[0],
    )


def exercise(model, inputs, metadata, synchronize, deadline):
    outputs, observations = {}, {}
    for case in CASES:
        for name, ids, mask, cap in requests(case, inputs, metadata["source_lengths"]):
            print("ENVELOPE_REQUEST", case, name, "cap", cap, flush=True)
            key = case + "__" + name
            outputs[key], observations[key] = observe_request(
                model, ids, mask, metadata["target_id"], cap, synchronize, deadline
            )
    checks = behavior_checks(outputs, observations, metadata["target_id"], model.config)
    return outputs, dict(checks=checks, observations=observations, same_tt_passed=all(checks.values()))


@contextmanager
def learned_compute_guard(torch, ttnn):
    """Allow token/control CPU work; reject learned CPU inference during generation."""
    from torch.utils._python_dispatch import TorchDispatchMode

    forbidden = (
        "aten.mm.",
        "aten.bmm.",
        "aten.addmm.",
        "aten.matmul.",
        "aten.linear.",
        "aten.embedding.",
        "aten.native_layer_norm.",
        "aten.layer_norm.",
        "aten._softmax.",
        "aten.softmax.",
        "aten.gelu.",
        "aten.scaled_dot_product",
        "aten._scaled_dot_product",
    )

    class Guard(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if str(func).startswith(forbidden):
                raise AssertionError("learned CPU compute: " + str(func))
            return func(*args, **(kwargs or {}))

    originals, counter = {}, [0]

    def counted(function):
        def call(*args, **kwargs):
            result = function(*args, **kwargs)
            counter[0] += 1
            return result

        return call

    try:
        for name in ("matmul", "linear", "embedding", "layer_norm", "execute_trace"):
            if hasattr(ttnn, name):
                originals[name] = getattr(ttnn, name)
                setattr(ttnn, name, counted(originals[name]))
        with Guard():
            yield counter
    finally:
        for name, function in originals.items():
            setattr(ttnn, name, function)


def weight_hashes(checkpoint):
    root = Path(checkpoint).resolve()
    if root.is_file():
        return {root.name: file_hash(root)}
    index = root / "pytorch_model.bin.index.json"
    names = (
        sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["pytorch_model.bin"]
    )
    if any(Path(name).name != name or not name.endswith(".bin") for name in names):
        raise ValueError("invalid checkpoint filenames")
    if any(not (root / name).resolve().is_relative_to(root) for name in names):
        raise ValueError("checkpoint shard resolves outside checkpoint directory")
    return {name: file_hash(root / name) for name in names}


def compare_oracle(path, pinned_sha, identity, outputs, target, config, precision="bf16"):
    if not pinned_sha or Path(path).stat().st_size > 1024 * 1024 or file_hash(path) != pinned_sha:
        raise ValueError("FP32 fixture must match an external SHA256 pin and size bound")
    fixture = json.loads(Path(path).read_text())
    if (
        fixture.get("schema") != "nllb-portable-envelope-fp32-v1"
        or fixture.get("identity") != identity
        or fixture.get("precision") != "fp32"
        or fixture.get("tf32") is not False
        or fixture.get("outputs", {}).keys() != outputs.keys()
    ):
        raise ValueError("FP32 fixture identity/precision/output keys mismatch")
    matches = {}
    for key, actual in outputs.items():
        cap = 63 if key.endswith("__cap63") else 64
        raw = fixture["outputs"][key]
        if not isinstance(raw, list) or not all(
            isinstance(row, list) and all(type(v) is int for v in row) for row in raw
        ):
            raise ValueError("FP32 fixture token arrays must contain integers")
        expected = np.asarray(raw, dtype=np.int64)
        matches[key] = canonical(actual, len(actual), target, cap, config) == canonical(
            expected, len(actual), target, cap, config
        )
    return dict(
        assessed=True,
        exact=all(matches.values()),
        per_request=matches,
        fixture_sha256=pinned_sha,
        candidate_precision=precision,
        reference_precision="fp32",
    )


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--tokenizer-directory")
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--oracle")
    parser.add_argument("--oracle-sha256")
    parser.add_argument("--precision", choices=("bf16", "bfp8_b"), default="bf16")
    parser.add_argument(
        "--timeout", type=float, default=280, help="soft total deadline; caller must impose a hard process timeout"
    )
    return parser


def write_report(path, report):
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def finalize(report, path, device, close_device, checks_passed, writer=write_report):
    """A durable green record requires successful checks, cleanup, and writes."""
    report.update(passed=False, device_closed=False)
    try:
        writer(path, report)
    except Exception as error:
        report["write_error"] = str(error)
    finally:
        if device is not None:
            try:
                close_device(device)
                report["device_closed"] = True
            except Exception as error:
                report["close_error"] = str(error)
    report["passed"] = bool(checks_passed and report["device_closed"] and "write_error" not in report)
    try:
        writer(path, report)
    except Exception as error:
        report.update(passed=False, write_error=str(error))
        # The atomic writer leaves the earlier failed-marked record intact.


def main(argv=None):
    args = argument_parser().parse_args(argv)
    if not 1 <= args.timeout <= 1180:
        raise ValueError("--timeout must be between 1 and 1180 seconds")
    deadline = time.monotonic() + args.timeout
    # Lazy device imports keep pytest collection and helper tests hardware-free.
    import torch
    import ttnn
    from transformers import AutoTokenizer

    from models.experimental.nllb.tt import backend
    from models.experimental.nllb.tt.nllb_validation import validate_config

    root = Path(args.checkpoint)
    asset_root = root if root.is_dir() else root.parent
    config_path = Path(args.config) if args.config else asset_root / "config.json"
    config = validate_config(json.loads(config_path.read_text()))
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_directory or str(asset_root), local_files_only=True, trust_remote_code=False, src_lang="eng_Latn"
    )
    inputs, metadata = build_inputs(tokenizer, config)
    identity = dict(
        config_sha256=file_hash(config_path),
        input_hashes={k: array_hash(v) for k, v in inputs.items()},
        target_id=metadata["target_id"],
        requests={case: [r[0] for r in requests(case, inputs, metadata["source_lengths"])] for case in CASES},
    )
    report = dict(
        passed=False,
        precision=args.precision,
        identity=identity,
        source_lengths=metadata["source_lengths"],
        fp32=dict(assessed=False, exact=None),
        scope="Trained generation behavior; not translation-quality or performance certification.",
    )
    device, checks_passed = None, False
    try:
        if bool(args.oracle) != bool(args.oracle_sha256):
            raise ValueError("provide both --oracle and --oracle-sha256")
        if args.oracle:
            identity["weight_sha256"] = weight_hashes(args.checkpoint)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = ttnn.open_device(device_id=args.device)
        device.enable_program_cache()
        model = backend.create_backend(args.checkpoint, config, device, precision=args.precision)
        report["effective_precision_policy"] = model.precision_policy
        if model.precision_policy.get("mode") != args.precision:
            raise AssertionError("unexpected effective precision")
        with learned_compute_guard(torch, ttnn) as calls:
            outputs, result = exercise(model, inputs, metadata, lambda: ttnn.synchronize_device(device), deadline)
        report.update(result, tt_calls=calls[0])
        if calls[0] <= 0:
            raise AssertionError("no observed TT learned operations")
        if args.oracle:
            report["fp32"] = compare_oracle(
                args.oracle, args.oracle_sha256, identity, outputs, metadata["target_id"], config, args.precision
            )
        checks_passed = result["same_tt_passed"] and report["fp32"].get("exact") is not False
    except Exception as error:
        report["error"] = type(error).__name__ + ": " + str(error)
    finally:
        finalize(report, args.output, device, ttnn.close_device, checks_passed)
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
