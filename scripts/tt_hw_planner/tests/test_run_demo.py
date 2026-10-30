# SPDX-License-Identifier: Apache-2.0
"""run-demo: the discovery + collection logic is generic (no model/stage names), driven by what the
demo declares and writes. These tests fix that contract without needing a device."""

from pathlib import Path

from tt_hw_planner.commands import run_demo as rd


def test_kind_is_keyed_on_extension_not_on_a_model_name():
    assert rd._kind_of(Path("a/sample_00.wav")) == "audio"
    assert rd._kind_of(Path("a/edit_3.PNG")) == "image"
    assert rd._kind_of(Path("a/answers.txt")) == "text"
    assert rd._kind_of(Path("a/weights.bin")) is None


def _demo(tmp_path, name, body):
    d = tmp_path / "demo"
    d.mkdir(parents=True, exist_ok=True)
    f = d / name
    f.write_text(body)
    return f


def test_flags_are_read_statically_from_the_demos_own_argparse(tmp_path):
    f = _demo(
        tmp_path,
        "demo_tts.py",
        "import argparse\n"
        "def main():\n"
        "    p = argparse.ArgumentParser()\n"
        "    p.add_argument('--out-dir', default='/tmp/x')\n"
        "    p.add_argument('--batch', type=int, default=32)\n"
        "if __name__ == '__main__':\n    main()\n",
    )
    out_flag, out_default, batch_flag = rd._discover_flags(f)
    assert out_flag == "--out-dir"
    assert out_default == "/tmp/x"
    assert batch_flag == "--batch"


def test_the_demo_matching_the_runs_task_is_chosen_over_the_others(tmp_path):
    for n in ("demo.py", "demo_text_continuation.py", "demo_text_to_speech.py"):
        _demo(tmp_path, n, "def main():\n    pass\nif __name__=='__main__':\n    main()\n")
    _file, mod = rd._pick_demo(tmp_path, "mymodel", ["text", "speech"])
    assert mod == "models.demos.mymodel.demo.demo_text_to_speech"
    # With no task hint, the first runnable demo is used (deterministic, sorted).
    _f2, mod2 = rd._pick_demo(tmp_path, "mymodel", [])
    assert mod2 == "models.demos.mymodel.demo.demo"


def test_media_files_decide_modality_and_text_falls_back_to_stdout(tmp_path):
    (tmp_path / "sample_00.wav").write_bytes(b"RIFF")
    (tmp_path / "sample_01.wav").write_bytes(b"RIFF")
    modality, items = rd._collect(tmp_path, "ignored stdout")
    assert modality == "audio"
    assert len(items) == 2 and all(i["kind"] == "audio" for i in items)

    empty = tmp_path / "empty"
    empty.mkdir()
    modality2, items2 = rd._collect(empty, "the batch of answers")
    assert modality2 == "text"
    assert items2 and items2[0]["text"] == "the batch of answers"
