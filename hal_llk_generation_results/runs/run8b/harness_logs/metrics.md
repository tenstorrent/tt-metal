_metrics.py version 2_

| Metric | 20261121_205834_B_c0e38f1f_claude-opus-5-5 |
|---|---|
| Duration (min) | 223.2 |
|   of which model time (min) | 74.4 |
| Number of steps (turns) | 447 |
| Input tokens (uncached) | 538 |
| Input tokens written to cache | 1.245.738 |
| Input tokens read from cache | 151.324.472 |
| Output tokens | 460.131 |
| Cache share of input | 0.992 |
| Cost (USD) | 48.44 |
| Bash commands with pytest | 160 |
| Pytest calls in them | 201 |
|   on the card | 185 |
|   compile only | 0 |
|   compile error | 4 |
|   wrong results | 46 |
|   hang / timeout | 35 |
|   pass | 78 |
|   denied | 5 |
| Iterations (agent test) | 147 |
| First pass of any selection (command no.) | 36 |
| Card resets (tt-smi -r) | 48 |
| File edits (Edit / Write) | 188 / 12 |
| File reads (Read) | 66 |
| Helper agents | 1 |
| Confluence / MCP calls | 0 |
| Denied tool calls | 29 |

### Flow of pytest invocations

| # | Call | Outcome | Passed | Failed | Selection | Command |
|---|---|---|---|---|---|---|
| 1 | 1 | denied | — | — | full | `cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261121_205834_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && which pytest; echo $CHIP_ARCH; ls -a` |
| 2 | 1 | collect | — | — | full | `cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261121_205834_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && pytest python_tests/test_eltwise_binary` |
| 3 | 1 | denied | — | — | full | `pytest python_tests/test_eltwise_binary_hal.py -k 'add' --collect-only -q 2>&1 \| grep 'test_eltwise_binary\[' \| grep 'None_' \| grep 'transpose_srca:No' \| grep '` |
| 4 | 1 | collect | — | — | subset:-k | `pytest python_tests/test_eltwise_binary_hal.py -k 'add and None_ and not bfp4 and not reuse and not int8 and not Bfp and not Float32 and not 16]' --collect-only` |
| 5 | 1 | compile_error | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 6 | 1 | hang | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 7 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 8 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 9 | 1 | compile_error | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 10 | 1 | hang | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 11 | 1 | hang | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 12 | 1 | hang | 0 | 1 | subset:nodeids | `timeout 600 pytest -x --logging-level=debug 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_typ` |
| 13 | 1 | other | — | — | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261121_205834_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 p` |
| 14 | 1 | other | — | — | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261121_205834_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 p` |
| 15 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261121_205834_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 p` |
| 16 | 1 | other | — | — | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261121_205834_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 p` |
| 17 | 1 | other | — | — | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261121_205834_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 p` |
| 18 | 1 | other | — | — | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261121_205834_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 p` |
| 19 | 1 | other | — | — | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261121_205834_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 p` |
| 20 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261121_205834_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 p` |
| 21 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261121_205834_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 p` |
| 22 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261121_205834_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 p` |
| 23 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261121_205834_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 p` |
| 24 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261121_205834_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 p` |
| 25 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261121_205834_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 p` |
| 26 | 1 | hang | — | — | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261121_205834_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 p` |
| 27 | 1 | hang | — | — | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261121_205834_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 p` |
| 28 | 1 | hang | — | — | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261121_205834_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 p` |
| 29 | 1 | hang | — | — | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261121_205834_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 p` |
| 30 | 1 | hang | — | — | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261121_205834_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 p` |
| 31 | 1 | help | — | — | full | `pytest python_tests --help 2>&1 \| tail -90 \| head -70` |
| 32 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261121_205834_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 p` |
| 33 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261121_205834_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 p` |
| 34 | 1 | other | — | — | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261121_205834_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 p` |
| 35 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261121_205834_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 p` |
| 36 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261121_205834_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 p` |
| 37 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 38 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-mat` |
| 38 | 2 | hang | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-mat` |
| 39 | 1 | hang | 1 | 1 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261121_205834_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 p` |
| 40 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_typ` |
| 40 | 2 | hang | 0 | 1 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_typ` |
| 41 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_typ` |
| 41 | 2 | hang | 0 | 1 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_typ` |
| 42 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_typ` |
| 42 | 2 | hang | 0 | 1 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_typ` |
| 43 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_typ` |
| 43 | 2 | hang | 0 | 1 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_typ` |
| 44 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_typ` |
| 44 | 2 | fail | 0 | 1 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_typ` |
| 45 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_typ` |
| 45 | 2 | hang | 0 | 1 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_typ` |
| 46 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_typ` |
| 46 | 2 | hang | 0 | 1 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_typ` |
| 47 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_typ` |
| 47 | 2 | hang | 0 | 1 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_typ` |
| 48 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast` |
| 49 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast` |
| 50 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast` |
| 51 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast` |
| 52 | 1 | fail | 0 | 1 | subset:nodeids | `rm ../tests/hal_llk/debug_dump.h; tt-smi -r 2>&1 \| tail -1; timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-form` |
| 53 | 1 | denied | — | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 54 | 1 | denied | — | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 55 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 56 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 57 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 58 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-mat` |
| 58 | 2 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-mat` |
| 59 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-mat` |
| 59 | 2 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-mat` |
| 60 | 1 | denied | — | — | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; for t in test_eltwise_binary test_eltwise_binary_hal test_eltwise_binary; do timeout 600 pytest -x "python_tests/$t.py::test_eltwise_b` |
| 61 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_typ` |
| 61 | 2 | fail | 0 | 1 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_typ` |
| 61 | 3 | pass | 1 | 0 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_typ` |
| 62 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 63 | 1 | hang | — | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 64 | 1 | fail | — | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 65 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-mat` |
| 65 | 2 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-mat` |
| 65 | 3 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-mat` |
| 66 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-mat` |
| 66 | 2 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-mat` |
| 67 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-mat` |
| 67 | 2 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-mat` |
| 68 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-mat` |
| 68 | 2 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-mat` |
| 69 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 70 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 71 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 72 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 73 | 1 | compile_error | — | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 74 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 75 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 76 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 77 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 78 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 79 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 80 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 81 | 1 | hang | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 82 | 1 | hang | — | — | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast` |
| 83 | 1 | other | — | — | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast` |
| 84 | 1 | fail | — | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 85 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 86 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 87 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 88 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-mat` |
| 88 | 2 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-mat` |
| 89 | 1 | pass | 1 | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 90 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 90 | 2 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 91 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 91 | 2 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 92 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 92 | 2 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 93 | 1 | hang | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 93 | 2 | hang | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 94 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 94 | 2 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 95 | 1 | pass | 1 | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 96 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 96 | 2 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 97 | 1 | pass | 1 | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 98 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 98 | 2 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 99 | 1 | pass | 1 | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 100 | 1 | pass | 1 | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 101 | 1 | pass | 1 | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 102 | 1 | pass | 1 | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 103 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 103 | 2 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 104 | 1 | pass | 1 | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 105 | 1 | pass | 1 | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 106 | 1 | collect | — | — | subset:-k | `pytest python_tests/test_eltwise_binary_hal.py -k 'bfp4 and add and None_ and Float16_b and dest_acc:No and 32, 32] and not 64' --collect-only -q 2>&1 \| head -8` |
| 107 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_bfp4_b[dest_acc:No-formats:Bfp4_b->Float16_b-broadcast_type:None_-math_fidel` |
| 108 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_bfp4_b[dest_acc:No-formats:Bfp4_b->Float16_b-broadcast_type:None_-math_fidel` |
| 109 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_bfp4_b[dest_acc:No-formats:Bfp4_b->Float16_b-broadcast_type:None_-math_fidel` |
| 109 | 2 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_bfp4_b[dest_acc:No-formats:Bfp4_b->Float16_b-broadcast_type:None_-math_fidel` |
| 110 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 110 | 2 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 111 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 111 | 2 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 112 | 1 | help | — | — | full | `pytest python_tests --help 2>&1 \| tr -s ' ' \| head -230 \| tail -90` |
| 113 | 1 | help | — | — | full | `pytest python_tests --help 2>&1 \| tr -s ' ' \| head -330 \| tail -100` |
| 114 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 115 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 116 | 1 | empty | 0 | 0 | subset:-k | `timeout 1500 pytest python_tests/test_eltwise_binary_hal.py -k 'test_eltwise_binary[ and Elwadd and None_ and transpose_srca:No and 32, 32]]' 2>&1 \| tr -d '\033` |
| 117 | 1 | fail | 5 | 19 | subset:-k | `timeout 1500 pytest python_tests/test_eltwise_binary_hal.py -k 'Elwadd and None_ and not bfp4 and not reuse and not int8 and not Yes and formats:Float16_b' -q 2` |
| 118 | 1 | fail | 5 | 19 | subset:-k | `timeout 1500 pytest python_tests/test_eltwise_binary_hal.py -k 'Elwadd and None_ and not bfp4 and not reuse and not int8 and not Yes and formats:Float16_b' -q -` |
| 119 | 1 | fail | 5 | 19 | subset:-k | `timeout 1500 pytest python_tests/test_eltwise_binary_hal.py -k 'Elwadd and None_ and not bfp4 and not reuse and not int8 and not Yes and formats:Float16_b' -q -` |
| 120 | 1 | other | — | — | subset:-k | `timeout 1500 pytest python_tests/test_eltwise_binary_hal.py -k 'Elwadd and None_ and not bfp4 and not reuse and not int8 and not Yes and formats:Float16_b' -v -` |
| 121 | 1 | fail | 5 | 19 | subset:-k | `timeout 1500 pytest python_tests/test_eltwise_binary_hal.py -k 'Elwadd and None_ and not bfp4 and not reuse and not int8 and not Yes and formats:Float16_b' -q -` |
| 122 | 1 | fail | 2 | 22 | subset:-k | `timeout 1500 pytest python_tests/test_eltwise_binary_hal.py -k 'Elwadd and None_ and not bfp4 and not reuse and not int8 and not Yes and formats:Float16_b' -q -` |
| 123 | 1 | pass | 1 | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 124 | 1 | pass | 1 | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 125 | 1 | pass | 1 | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 126 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 126 | 2 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 127 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 128 | 1 | pass | 1 | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 129 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest -x --logging-level=debug 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_typ` |
| 130 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x --logging-level=debug 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_typ` |
| 131 | 1 | pass | 1 | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 132 | 1 | pass | 1 | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 133 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 133 | 2 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 134 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 134 | 2 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 135 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 135 | 2 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 136 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 136 | 2 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 137 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 137 | 2 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 138 | 1 | other | — | — | subset:-k | `timeout 1500 pytest python_tests/test_eltwise_binary_hal.py -k 'Elwadd and None_ and not bfp4 and not reuse and not int8 and not Yes and formats:Float16_b' -v -` |
| 139 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 140 | 1 | compile_error | — | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 141 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 141 | 2 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 142 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 142 | 2 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 143 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 143 | 2 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwsub-mat` |
| 144 | 1 | pass | 72 | 0 | subset:-k | `timeout 3000 pytest python_tests/test_eltwise_binary_hal.py -k 'Elwadd and None_ and not bfp4 and not reuse and not int8 and not Yes' -q --tb=no --show-capture=` |
| 145 | 1 | fail | 1.596 | 64 | full | `timeout 3500 pytest python_tests/test_eltwise_binary_hal.py -k 'add' -q --tb=no --show-capture=no -o log_cli=false -rf 2>&1 \| tr -d '\033' \| tail -40 \| cut -c 1` |
| 146 | 1 | other | — | — | subset:--lf | `timeout 3500 pytest python_tests/test_eltwise_binary_hal.py -k 'add' -q --tb=no --show-capture=no -o log_cli=false -rf --lf 2>&1 \| tr -d '\033' \| tail -66 \| cut` |
| 147 | 1 | other | — | — | subset:--lf | `timeout 3500 pytest python_tests/test_eltwise_binary_hal.py -k 'add' --tb=no --show-capture=no -o log_cli=false --lf -v 2>&1 \| tr -d '\033' \| tail -68 \| cut -c ` |
| 148 | 1 | pass | 40 | 0 | subset:--lf | `timeout 3500 pytest python_tests/test_eltwise_binary_hal.py -k 'add and not reuse' --lf -q --tb=no --show-capture=no -o log_cli=false 2>&1 \| tr -d '\033' \| tail` |
| 149 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest -x 'python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_dest_reuse[reuse_dest_type:DEST_TO_SRCB-math_op:Elwadd-formats:Float16_b->Fl` |
| 150 | 1 | other | — | — | subset:-k | `timeout 3500 pytest python_tests/test_eltwise_binary_hal.py -k 'add and reuse' -v --tb=no --show-capture=no -o log_cli=false 2>&1 \| tr -d '\033' \| tail -26 \| cu` |
| 151 | 1 | pass | 1.660 | 0 | full | `tt-smi -r 2>&1 \| tail -1 && timeout 3500 pytest python_tests/test_eltwise_binary_hal.py -k 'add' -q --tb=no --show-capture=no -o log_cli=false 2>&1 \| tr -d '\03` |
| 152 | 1 | pass | 1.660 | 0 | full | `timeout 3500 pytest python_tests/test_eltwise_binary.py -k 'add' -q --tb=no --show-capture=no -o log_cli=false 2>&1 \| tr -d '\033' \| tail -1` |
| 153 | 1 | pass | 1.660 | 0 | full | `timeout 3500 pytest python_tests/test_eltwise_binary_hal.py -k 'add' -q --tb=no --show-capture=no -o log_cli=false 2>&1 \| tr -d '\033' \| tail -1` |
| 154 | 1 | pass | 1.660 | 0 | full | `tt-smi -r 2>&1 \| tail -1 && timeout 3500 pytest python_tests/test_eltwise_binary_hal.py -k 'add' -q --tb=no --show-capture=no -o log_cli=false 2>&1 \| tr -d '\03` |
| 155 | 1 | pass | 1.660 | 0 | full | `timeout 3500 pytest python_tests/test_eltwise_binary.py -k 'Elwsub' -q --tb=no --show-capture=no -o log_cli=false 2>&1 \| tr -d '\033' \| tail -1; timeout 3500 py` |
| 155 | 2 | pass | 1.660 | 0 | full | `timeout 3500 pytest python_tests/test_eltwise_binary.py -k 'Elwsub' -q --tb=no --show-capture=no -o log_cli=false 2>&1 \| tr -d '\033' \| tail -1; timeout 3500 py` |
| 156 | 1 | pass | 1.660 | 0 | full | `timeout 3500 pytest python_tests/test_eltwise_binary.py -k 'Elwsub' -q --tb=no --show-capture=no -o log_cli=false 2>&1 \| tr -d '\033' \| tail -1; timeout 3500 py` |
| 156 | 2 | pass | 1.660 | 0 | full | `timeout 3500 pytest python_tests/test_eltwise_binary.py -k 'Elwsub' -q --tb=no --show-capture=no -o log_cli=false 2>&1 \| tr -d '\033' \| tail -1; timeout 3500 py` |
| 157 | 1 | pass | 1.660 | 0 | full | `tt-smi -r 2>&1 \| tail -1 && timeout 3500 pytest python_tests/test_eltwise_binary_hal.py -k 'add' -q --tb=no --show-capture=no -o log_cli=false 2>&1 \| tr -d '\03` |
| 158 | 1 | pass | 1.660 | 0 | full | `timeout 3500 pytest python_tests/test_eltwise_binary.py -k 'add' -q --tb=no --show-capture=no -o log_cli=false 2>&1 \| tr -d '\033' \| tail -1` |
| 159 | 1 | pass | 1.660 | 0 | full | `timeout 3500 pytest python_tests/test_eltwise_binary_hal.py -k 'add' -q --tb=no --show-capture=no -o log_cli=false 2>&1 \| tr -d '\033' \| tail -1` |
| 160 | 1 | pass | 1.660 | 0 | full | `tt-smi -r 2>&1 \| tail -1 && timeout 3500 pytest python_tests/test_eltwise_binary_hal.py -k 'add' -q --tb=no --show-capture=no -o log_cli=false 2>&1 \| tr -d '\03` |

_The outcome classification is heuristic (based on the pytest output) and is meant as a guide._

### First full pass per test

| Test | Iteration (total / of that test) | Command no. | Time (min) | Input tokens | Original passed |
|---|---|---|---|---|---|
| test_eltwise_binary_hal | 140 / 140 | 151 | 186.89 | 140.596.374 | 1.660 |

### First full pass of the final code (after the last change of the agent's code)

| Test | Iteration (total / of that test) | Command no. | Time (min) | Input tokens |
|---|---|---|---|---|
| test_eltwise_binary_hal | 144 / 144 | 156 | 209.01 | 145.183.407 |

Iterations total: 147; all tests fully passed at iteration: 140; changes of the agent's code: 200; ambiguous commands (the number of pytest calls and final lines does not match): 18

| Test | Success@5 | Success@10 | Success@15 | Success@20 | Success@25 | Success@50 |
|---|---|---|---|---|---|---|
| test_eltwise_binary_hal | no | no | no | no | no | no |

### Cost by iteration outcome kind

| Outcome | Iterations | Minutes | Input tokens |
|---|---|---|---|
| compile_error | 4 | 23.51 | 18.429.146 |
| fail | 45 | 57.51 | 45.504.532 |
| hang | 34 | 46.43 | 30.051.667 |
| other | 26 | 24.89 | 20.379.418 |
| pass | 38 | 69.9 | 34.509.621 |

### Reading of material

| Category | Calls | Distinct | Re-reads | Share of all tool calls |
|---|---|---|---|---|
| hal_docs | 39 | 20 | 19 | 0.083 |
| raw_ref | 0 | 0 | 0 | 0.0 |
| spec | 19 | 19 | 0 | 0.041 |
| tests | 11 | 11 | 0 | 0.023 |
| own_code | 7 | 3 | 4 | 0.015 |

_Iteration = a run of the agent's (`*_hal`) test, including a compile failure. Full pass = the same selection of variants as the original (same -k, the whole test file, or the same number of selected and deselected variants), pass count >= original, 0 failed. Input tokens up to the first pass are the sum of `usage` from the transcript messages up to that moment; output tokens cannot be counted reliably per moment from the transcript, so they are given only in total (from the `result` event)._
