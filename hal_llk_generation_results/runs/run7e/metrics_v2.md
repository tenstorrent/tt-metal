_metrics.py version 2_

| Metric | kernel_run7e |
|---|---|
| Duration (min) | 105.4 |
|   of which model time (min) | 52.5 |
| Number of steps (turns) | 350 |
| Input tokens (uncached) | 774 |
| Input tokens written to cache | 1.248.879 |
| Input tokens read from cache | 98.664.248 |
| Output tokens | 335.403 |
| Cache share of input | 0.987 |
| Cost (USD) | 33.56 |
| Bash commands with pytest | 88 |
| Pytest calls in them | 91 |
|   on the card | 81 |
|   compile only | 1 |
|   compile error | 3 |
|   wrong results | 42 |
|   hang / timeout | 16 |
|   pass | 14 |
|   denied | 5 |
| Iterations (agent test) | 81 |
| First pass of any selection (command no.) | 24 |
| Card resets (tt-smi -r) | 23 |
| File edits (Edit / Write) | 91 / 60 |
| File reads (Read) | 88 |
| Helper agents | 4 |
| Confluence / MCP calls | 0 |
| Denied tool calls | 47 |

### Flow of pytest invocations

| # | Call | Outcome | Passed | Failed | Selection | Command |
|---|---|---|---|---|---|---|
| 1 | 1 | collect | — | — | full | `cd /proj_sw/user_dev/ctr-sbratic/hal-runs/20261106_095649_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && pytest python_tests/test_eltwise_binary_hal.py ` |
| 2 | 1 | compile_error | 0 | 1 | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 3 | 1 | denied | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 4 | 1 | denied | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 5 | 1 | hang | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 6 | 1 | hang | 0 | 1 | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 7 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 8 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 9 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 10 | 1 | compile_error | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 11 | 1 | hang | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 12 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 13 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 14 | 1 | denied | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 15 | 1 | other | — | — | subset:nodeids | `NO_COLOR=1 timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:N` |
| 16 | 1 | denied | — | — | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 17 | 1 | hang | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 18 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r > /dev/null 2>&1; cd /proj_sw/user_dev/ctr-sbratic/hal-runs/20261106_095649_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 pytest ` |
| 19 | 1 | hang | — | — | subset:nodeids | `tt-smi -r > /dev/null 2>&1; cd /proj_sw/user_dev/ctr-sbratic/hal-runs/20261106_095649_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 pytest ` |
| 20 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r > /dev/null 2>&1; cd /proj_sw/user_dev/ctr-sbratic/hal-runs/20261106_095649_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 pytest ` |
| 21 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r > /dev/null 2>&1; cd /proj_sw/user_dev/ctr-sbratic/hal-runs/20261106_095649_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 pytest ` |
| 22 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r > /dev/null 2>&1; cd /proj_sw/user_dev/ctr-sbratic/hal-runs/20261106_095649_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 pytest ` |
| 23 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r > /dev/null 2>&1; cd /proj_sw/user_dev/ctr-sbratic/hal-runs/20261106_095649_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 pytest ` |
| 24 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r > /dev/null 2>&1; cd /proj_sw/user_dev/ctr-sbratic/hal-runs/20261106_095649_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 pytest ` |
| 24 | 2 | hang | 0 | 1 | subset:nodeids | `tt-smi -r > /dev/null 2>&1; cd /proj_sw/user_dev/ctr-sbratic/hal-runs/20261106_095649_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 pytest ` |
| 25 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r > /dev/null 2>&1; cd /proj_sw/user_dev/ctr-sbratic/hal-runs/20261106_095649_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 pytest ` |
| 25 | 2 | hang | 0 | 1 | subset:nodeids | `tt-smi -r > /dev/null 2>&1; cd /proj_sw/user_dev/ctr-sbratic/hal-runs/20261106_095649_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 pytest ` |
| 26 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > /dev/null 2>&1; cd /proj_sw/user_dev/ctr-sbratic/hal-runs/20261106_095649_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 pytest ` |
| 27 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 28 | 1 | hang | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 29 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r > /dev/null 2>&1; cd /proj_sw/user_dev/ctr-sbratic/hal-runs/20261106_095649_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 pytest ` |
| 30 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > /dev/null 2>&1; cd /proj_sw/user_dev/ctr-sbratic/hal-runs/20261106_095649_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 pytest ` |
| 31 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 32 | 1 | compile_error | — | — | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 33 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 34 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 35 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 36 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 37 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 38 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 39 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 40 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 41 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 42 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 43 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 44 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 45 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 46 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 47 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 48 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 49 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 50 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 51 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 52 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 53 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 54 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 55 | 1 | denied | — | — | subset:nodeids | `tt-smi -r > /dev/null 2>&1; cd /proj_sw/user_dev/ctr-sbratic/hal-runs/20261106_095649_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 pytest ` |
| 56 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 57 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 58 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 59 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 60 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 61 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 62 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 63 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 64 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 65 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 66 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 67 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 68 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_o` |
| 69 | 1 | other | — | — | subset:nodeids | `tt-smi -r > /dev/null 2>&1; cd /proj_sw/user_dev/ctr-sbratic/hal-runs/20261106_095649_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 pytest ` |
| 70 | 1 | other | — | — | subset:nodeids | `tt-smi -r > /dev/null 2>&1; cd /proj_sw/user_dev/ctr-sbratic/hal-runs/20261106_095649_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 pytest ` |
| 71 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r > /dev/null 2>&1; cd /proj_sw/user_dev/ctr-sbratic/hal-runs/20261106_095649_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 pytest ` |
| 72 | 1 | compile_only | 1.660 | 0 | full | `nproc; cd /proj_sw/user_dev/ctr-sbratic/hal-runs/20261106_095649_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 3000 pytest --compile-producer -` |
| 73 | 1 | fail | 1.644 | 16 | full | `timeout 7000 pytest --compile-consumer python_tests/test_eltwise_binary_hal.py -k 'add' -q --tb=no -rf -p no:sugar 2>&1 \| cut -c1-250 > hal_llk/run1.log; tail -` |
| 74 | 1 | hang | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_type:None_-mat` |
| 74 | 2 | hang | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_type:None_-mat` |
| 75 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_type:None_-mat` |
| 76 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_type:None_-mat` |
| 77 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_type:None_-mat` |
| 78 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > /dev/null 2>&1; cd /proj_sw/user_dev/ctr-sbratic/hal-runs/20261106_095649_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 pytest ` |
| 79 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest --color=no "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_type:None_-mat` |
| 80 | 1 | pass | 16 | 0 | subset:-k | `timeout 900 pytest python_tests/test_eltwise_binary_hal.py -k 'add and int8' -q --tb=no -p no:sugar 2>&1 \| tail -2` |
| 81 | 1 | pass | 24 | 0 | subset:-k | `timeout 2000 pytest python_tests/test_eltwise_binary_hal.py -k 'add and dest_reuse' -q --tb=no -p no:sugar 2>&1 \| tail -2` |
| 82 | 1 | fail | 1.636 | 24 | full | `cd /proj_sw/user_dev/ctr-sbratic/hal-runs/20261106_095649_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 7000 pytest python_tests/test_eltwise_b` |
| 83 | 1 | fail | 0 | 24 | subset:-k | `timeout 2000 pytest python_tests/test_eltwise_binary_hal.py -k 'add and dest_reuse' -q --tb=no -p no:sugar 2>&1 \| tail -1` |
| 84 | 1 | pass | 24 | 0 | subset:-k | `timeout 2000 pytest python_tests/test_eltwise_binary_hal.py -k 'add and dest_reuse' -q --tb=no -p no:sugar 2>&1 \| tail -1` |
| 85 | 1 | pass | 1.660 | 0 | full | `tt-smi -r 2>&1 \| tail -1; cd /proj_sw/user_dev/ctr-sbratic/hal-runs/20261106_095649_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 7000 pytest p` |
| 86 | 1 | pass | 1.660 | 0 | full | `timeout 7000 pytest python_tests/test_eltwise_binary.py -k 'add' -q --tb=no -p no:sugar 2>&1 \| tail -1` |
| 87 | 1 | pass | 1.660 | 0 | full | `timeout 7000 pytest python_tests/test_eltwise_binary_hal.py -k 'add' -q --tb=no -p no:sugar 2>&1 \| tail -1` |
| 88 | 1 | pass | 1.660 | 0 | full | `tt-smi -r 2>&1 \| tail -1; cd /proj_sw/user_dev/ctr-sbratic/hal-runs/20261106_095649_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 7000 pytest p` |

_The outcome classification is heuristic (based on the pytest output) and is meant as a guide._

### First full pass per test

| Test | Iteration (total / of that test) | Command no. | Time (min) | Input tokens | Original passed |
|---|---|---|---|---|---|
| test_eltwise_binary_hal | 79 / 79 | 85 | 95.09 | 96.808.181 | 1.660 |

### First full pass of the final code (after the last change of the agent's code)

| Test | Iteration (total / of that test) | Command no. | Time (min) | Input tokens |
|---|---|---|---|---|
| test_eltwise_binary_hal | 79 / 79 | 85 | 95.09 | 96.808.181 |

Iterations total: 81; all tests fully passed at iteration: 79; changes of the agent's code: 153; ambiguous commands (the number of pytest calls and final lines does not match): 0

| Test | Success@5 | Success@10 | Success@15 | Success@20 | Success@25 | Success@50 |
|---|---|---|---|---|---|---|
| test_eltwise_binary_hal | no | no | no | no | no | no |

### Cost by iteration outcome kind

| Outcome | Iterations | Minutes | Input tokens |
|---|---|---|---|
| compile_error | 3 | 18.11 | 13.804.914 |
| fail | 42 | 45.56 | 54.558.628 |
| hang | 16 | 16.3 | 13.159.781 |
| other | 9 | 6.33 | 7.368.429 |
| pass | 11 | 17.5 | 10.081.733 |

### Reading of material

| Category | Calls | Distinct | Re-reads | Share of all tool calls |
|---|---|---|---|---|
| hal_docs | 37 | 18 | 19 | 0.09 |
| raw_ref | 0 | 0 | 0 | 0.0 |
| spec | 16 | 16 | 0 | 0.039 |
| tests | 23 | 18 | 5 | 0.056 |
| own_code | 39 | 23 | 16 | 0.094 |

_Iteration = a run of the agent's (`*_hal`) test, including a compile failure. Full pass = the same selection of variants as the original (same -k, the whole test file, or the same number of selected and deselected variants), pass count >= original, 0 failed. Input tokens up to the first pass are the sum of `usage` from the transcript messages up to that moment; output tokens cannot be counted reliably per moment from the transcript, so they are given only in total (from the `result` event)._
