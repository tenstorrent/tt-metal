"""Audit a stock-reproducer fault capture using only the standard library."""
import argparse
import json
from pathlib import Path


def audit(capture_path, fixture_path):
    from raw_archive import load_archive

    fixture = load_archive(fixture_path)
    capture = load_archive(capture_path)
    source = capture["source"].byte_values()
    expected = capture["expected"].byte_values()
    output = capture["output"].byte_values()
    assert source == fixture["source_raw"].byte_values()
    assert expected == fixture["expected_raw"].byte_values()
    assert len(capture["rereads"]) == 3
    assert all(x.byte_values() == output for x in capture["rereads"])
    assert len(output) == len(expected)
    indices = [i for i, (actual, wanted) in enumerate(zip(output, expected)) if actual != wanted]
    fault = capture["report"]
    assert indices and len(indices) == fault["changed_bytes"]
    examples = [
        dict(
            offset=i,
            expected=expected[i],
            actual=output[i],
            xor=expected[i] ^ output[i],
        )
        for i in indices[:16]
    ]
    assert examples == fault["examples"]
    assert fault["source_unchanged"] and fault["rereads_equal"] == [True] * 3
    return dict(
        passed=True,
        hardware_imported=False,
        iteration=fault["iteration"],
        changed_bytes=len(indices),
        examples=examples,
        source_intact=True,
        rereads_agree=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--fixture", type=Path, default=Path(__file__).with_name("55041-read-fixture-v86.pt.gz"))
    args = parser.parse_args()
    print(json.dumps(audit(args.capture, args.fixture), indent=2))
