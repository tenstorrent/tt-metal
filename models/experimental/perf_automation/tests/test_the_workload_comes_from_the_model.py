# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The measured workload is the model's, and anything the test decides for itself is named out loud.

BATCH BELONGS TO THE MODEL and the channel for it works: the tool sends TT_PERF_BATCH=0, meaning "ask
the pipeline", and build_pipeline answers with its own declared batch. What defeated it on
Voxtral-Mini-3B was a SECOND axis the tool knows nothing about -- the generated test defined

    PERF_AUDIO_STREAMS = int(os.environ.get("TT_PERF_AUDIO_STREAMS", "2"))

a variable that appears nowhere in this repo. The agent writing the test decided the heaviest
trimmable input was the clip count and picked 2. The pipeline was still BUILT for its declared batch
of 8 and then handed 2 clips, so prefill measured a quarter of the real workload -- 148.76 ms printed
beside a 7.48 ms full-batch theoretical, with nothing to say the two described different workloads.

Trimming input for a tracy run is legitimate; that is how a deep model stays under the profiler's
marker limit. Inventing the axis and never reporting it is not.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def test_an_invented_capped_count_is_detected():
    """The exact line from the real generated test."""
    from agent.perf_test_gen import invented_workload_vars

    src = 'PERF_AUDIO_STREAMS = int(os.environ.get("TT_PERF_AUDIO_STREAMS", "2"))\n'
    assert invented_workload_vars(src) == [("TT_PERF_AUDIO_STREAMS", 2)]


def test_the_tools_own_knobs_are_not_flagged():
    """ISL, OSL, batch and depth are the tool's to set; flagging them would be noise."""
    from agent.perf_test_gen import invented_workload_vars

    src = (
        'a = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))\n'
        'b = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))\n'
        'c = int(os.environ.get("TT_PERF_BATCH", "0"))\n'
    )
    assert invented_workload_vars(src) == []


def test_a_per_stage_depth_variable_is_not_flagged():
    """Those names come from PIPELINE_STAGES and the tool sets them."""
    from agent.perf_test_gen import invented_workload_vars

    src = 'x = int(os.environ.get("TT_PERF_ENCODE_LAYERS", "2"))\n'
    assert invented_workload_vars(src, stages=("encode", "prefill", "decode")) == []


def test_zero_means_ask_the_pipeline_and_is_not_a_cap():
    """0 is the documented way to say 'the model decides' -- the opposite of a capped literal."""
    from agent.perf_test_gen import invented_workload_vars

    assert invented_workload_vars('n = int(os.environ.get("TT_PERF_CLIPS", "0"))') == []
    assert invented_workload_vars('n = int(os.environ.get("TT_PERF_CLIPS", "4"))') == [("TT_PERF_CLIPS", 4)]


def test_the_shipped_defect_is_caught():
    """The line as it actually shipped, kept as a fixture rather than read from the live file.

    IT USED TO READ THE FILE, and named the variable it expected to find in it. Both halves were
    wrong. The file is a GENERATED perf test -- the tool rewrites it whenever the workload changes --
    so the assertion pinned one snapshot of a moving target, and by 2026-08-16 that model's test
    invented TT_PERF_FLUSH_EVERY instead: the detector was working perfectly and the test failed
    anyway. Worse, it failed only in a tree that HAS the model, because the missing-file branch
    returned early -- so the tool's own suite was green everywhere except the trees where preflight
    runs it, which is the one place a red suite stops a run.

    It also named a model. Nothing else in this file does, and the detector knows nothing about
    audio: it looks for a work-capping literal in an environment default, whatever the stage.

    The regression is real, so the LINE is kept -- with the name it shipped under -- and the
    dependency on a file that regenerates is not."""
    from agent.perf_test_gen import invented_workload_vars

    src = 'streams = int(os.environ.get("TT_PERF_AUDIO_STREAMS", "1"))\n'
    assert invented_workload_vars(src) == [("TT_PERF_AUDIO_STREAMS", 1)]


def test_the_run_reports_them():
    """The defect was silence: the variable existed, capped the work, and nothing read it back."""
    src = (_PA / "agent" / "before_loop.py").read_text()
    assert "invented_workload_vars" in src, "the run never surfaces an invented workload knob"
    i = src.index("invented_workload_vars")
    assert "the tool does not set these" in src[i : i + 900]


def test_the_generator_is_told_where_the_workload_comes_from():
    src = (_PA / "agent" / "perf_test_gen.py").read_text()
    assert "WORKLOAD SIZE comes from the model" in src
    assert "DERIVE it from the batch the pipeline was built with" in src
