_metrics.py version 2_

| Metric | 20261122_005225_B_c0e38f1f_claude-opus-5-5 |
|---|---|
| Duration (min) | 162.4 |
|   of which model time (min) | 65.6 |
| Number of steps (turns) | 382 |
| Input tokens (uncached) | 440 |
| Input tokens written to cache | 1.183.851 |
| Input tokens read from cache | 114.696.197 |
| Output tokens | 408.272 |
| Cache share of input | 0.99 |
| Cost (USD) | 39.6 |
| Bash commands with pytest | 98 |
| Pytest calls in them | 104 |
|   on the card | 78 |
|   compile only | 12 |
|   compile error | 7 |
|   wrong results | 35 |
|   hang / timeout | 12 |
|   pass | 26 |
|   denied | 3 |
| Iterations (agent test) | 83 |
| First pass of any selection (command no.) | 30 |
| Card resets (tt-smi -r) | 72 |
| File edits (Edit / Write) | 156 / 30 |
| File reads (Read) | 76 |
| Helper agents | 1 |
| Confluence / MCP calls | 0 |
| Denied tool calls | 21 |

### Flow of pytest invocations

| # | Call | Outcome | Passed | Failed | Selection | Command |
|---|---|---|---|---|---|---|
| 1 | 1 | denied | — | — | full | `cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261122_005225_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 pytest python_tests/test_el` |
| 2 | 1 | collect | — | — | full | `cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261122_005225_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 pytest python_tests/test_el` |
| 3 | 1 | compile_only | 0 | 0 | subset:-k | `timeout 900 pytest --compile-producer -x python_tests/test_eltwise_binary_hal.py -k 'test_eltwise_binary_int8_format and Elwadd and 32, 32] and tile_dimensions:` |
| 4 | 1 | compile_error | 0 | 1 | subset:nodeids | `timeout 900 pytest --compile-producer -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_typ` |
| 5 | 1 | denied | — | — | subset:nodeids | `timeout 900 pytest --compile-producer -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_typ` |
| 6 | 1 | compile_error | 0 | 1 | subset:nodeids | `timeout 900 pytest --compile-producer -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_typ` |
| 7 | 1 | compile_only | — | — | subset:nodeids | `timeout 900 pytest --compile-producer -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_typ` |
| 8 | 1 | denied | — | — | subset:nodeids | `timeout 900 pytest --compile-producer -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_typ` |
| 9 | 1 | compile_only | — | — | subset:nodeids | `timeout 900 pytest --compile-producer -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_typ` |
| 10 | 1 | compile_error | — | — | subset:nodeids | `timeout 900 pytest --compile-producer -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_typ` |
| 11 | 1 | compile_error | — | — | subset:nodeids | `timeout 900 pytest --compile-producer -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_typ` |
| 12 | 1 | compile_error | — | — | subset:nodeids | `timeout 900 pytest --compile-producer -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_typ` |
| 13 | 1 | compile_error | — | — | subset:nodeids | `timeout 900 pytest --compile-producer -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_typ` |
| 14 | 1 | compile_only | — | — | subset:nodeids | `timeout 900 pytest --compile-producer -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_typ` |
| 15 | 1 | compile_only | — | — | subset:nodeids | `timeout 900 pytest --compile-producer -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_typ` |
| 16 | 1 | compile_only | — | — | subset:nodeids | `timeout 900 pytest --compile-producer -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_typ` |
| 17 | 1 | compile_only | — | — | subset:nodeids | `timeout 900 pytest --compile-producer -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_typ` |
| 18 | 1 | collect | — | — | subset:-k | `timeout 900 pytest --collect-only -q python_tests/test_eltwise_binary_hal.py -k "add and Float16_b->Float16_b and None_ and 32, 32 and not dest_reuse and not bf` |
| 19 | 1 | collect | — | — | subset:-k | `timeout 900 pytest --collect-only -q python_tests/test_eltwise_binary_hal.py -k "add and Elwmul" 2>&1 \| tail -2; timeout 900 pytest --collect-only -q python_tes` |
| 20 | 1 | hang | 0 | 1 | subset:nodeids | `timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 21 | 1 | hang | 0 | 1 | subset:nodeids | `timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd` |
| 22 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 23 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 24 | 1 | compile_error | 0 | 1 | subset:nodeids | `timeout 900 pytest --compile-producer -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_typ` |
| 25 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 26 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 27 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 28 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 29 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 30 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-bro` |
| 31 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 32 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 33 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 34 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 35 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 36 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 37 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 38 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 39 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 40 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 41 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 42 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 43 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 44 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 45 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 46 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 47 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 48 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 49 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 50 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x --logging-level=debug "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-format` |
| 51 | 1 | other | — | — | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x --logging-level=debug "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:Yes-forma` |
| 52 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x --logging-level=debug "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-format` |
| 52 | 2 | pass | 1 | 0 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x --logging-level=debug "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-format` |
| 53 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 54 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 55 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 56 | 1 | other | — | — | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x --logging-level=debug "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-format` |
| 57 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 58 | 1 | other | — | — | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x --logging-level=debug "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-format` |
| 59 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 60 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 61 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 62 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 63 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 64 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 65 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 66 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 67 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 68 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 69 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 70 | 1 | other | — | — | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 71 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 72 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x --logging-level=debug "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-format` |
| 73 | 1 | other | — | — | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x --logging-level=debug "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-format` |
| 74 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 75 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 76 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 77 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 78 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 79 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 80 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 81 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 82 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 83 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 83 | 2 | pass | 1 | 0 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 84 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 84 | 2 | pass | 1 | 0 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 85 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 85 | 2 | pass | 1 | 0 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b` |
| 86 | 1 | compile_only | 1.660 | 0 | full | `nproc; timeout 3000 pytest --compile-producer -n 32 -q python_tests/test_eltwise_binary_hal.py -k add > hal_llk/compile.txt 2>&1; tail -5 hal_llk/compile.txt` |
| 87 | 1 | fail | 1.604 | 56 | full | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 5400 pytest --compile-consumer -q python_tests/test_eltwise_binary_hal.py -k add > hal_llk/run.txt 2>&1; tail -3 hal` |
| 88 | 1 | collect | — | — | subset:--lf | `timeout 600 pytest --lf --collect-only -q python_tests/test_eltwise_binary_hal.py -k add 2>&1 \| cut -d'[' -f2 \| cut -c 1-170 \| head -60` |
| 89 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x --logging-level=debug "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_a` |
| 90 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r > hal_llk/reset.txt 2>&1; timeout 900 pytest -x --logging-level=debug "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_a` |
| 91 | 1 | compile_only | 1.660 | 0 | full | `timeout 3000 pytest --compile-producer -n 24 -q python_tests/test_eltwise_binary_hal.py -k add > hal_llk/compile.txt 2>&1; tail -1 hal_llk/compile.txt; tt-smi -` |
| 91 | 2 | pass | 1.660 | 0 | full | `timeout 3000 pytest --compile-producer -n 24 -q python_tests/test_eltwise_binary_hal.py -k add > hal_llk/compile.txt 2>&1; tail -1 hal_llk/compile.txt; tt-smi -` |
| 92 | 1 | compile_only | 1.660 | 0 | full | `timeout 3000 pytest --compile-producer -n 24 -q python_tests/test_eltwise_binary_hal.py -k add > hal_llk/compile.txt 2>&1; tail -1 hal_llk/compile.txt; tt-smi -` |
| 92 | 2 | pass | 1.660 | 0 | full | `timeout 3000 pytest --compile-producer -n 24 -q python_tests/test_eltwise_binary_hal.py -k add > hal_llk/compile.txt 2>&1; tail -1 hal_llk/compile.txt; tt-smi -` |
| 93 | 1 | compile_only | 1.660 | 0 | full | `timeout 3000 pytest --compile-producer -n 24 -q python_tests/test_eltwise_binary.py -k add > hal_llk/compile_orig.txt 2>&1; tail -1 hal_llk/compile_orig.txt` |
| 94 | 1 | pass | 1.660 | 0 | full | `tt-smi -r > hal_llk/reset.txt 2>&1; tail -1 hal_llk/reset.txt; timeout 7000 pytest python_tests/test_eltwise_binary_hal.py -k 'add' > hal_llk/seq1.txt 2>&1; tai` |
| 95 | 1 | pass | 1.660 | 0 | full | `timeout 7000 pytest python_tests/test_eltwise_binary.py -k 'add' > hal_llk/seq2.txt 2>&1; tail -1 hal_llk/seq2.txt` |
| 96 | 1 | pass | 1.660 | 0 | full | `timeout 7000 pytest python_tests/test_eltwise_binary_hal.py -k 'add' > hal_llk/seq3.txt 2>&1; tail -1 hal_llk/seq3.txt` |
| 97 | 1 | pass | 1.660 | 0 | full | `tt-smi -r > hal_llk/reset.txt 2>&1; tail -1 hal_llk/reset.txt; timeout 7000 pytest python_tests/test_eltwise_binary_hal.py -k 'add' > hal_llk/seq4.txt 2>&1; tai` |
| 98 | 1 | compile_only | 1 | 0 | subset:nodeids | `timeout 900 pytest --compile-producer -q "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:N` |

_The outcome classification is heuristic (based on the pytest output) and is meant as a guide._

### First full pass per test

| Test | Iteration (total / of that test) | Command no. | Time (min) | Input tokens | Original passed |
|---|---|---|---|---|---|
| test_eltwise_binary_hal | 79 / 79 | 91 | 139.52 | 108.108.502 | 1.660 |

### First full pass of the final code (after the last change of the agent's code)

| Test | Iteration (total / of that test) | Command no. | Time (min) | Input tokens |
|---|---|---|---|---|

Iterations total: 83; all tests fully passed at iteration: 79; changes of the agent's code: 185; ambiguous commands (the number of pytest calls and final lines does not match): 0

| Test | Success@5 | Success@10 | Success@15 | Success@20 | Success@25 | Success@50 |
|---|---|---|---|---|---|---|
| test_eltwise_binary_hal | no | no | no | no | no | no |

### Cost by iteration outcome kind

| Outcome | Iterations | Minutes | Input tokens |
|---|---|---|---|
| compile_error | 7 | 14.58 | 16.668.210 |
| fail | 35 | 54.44 | 39.417.438 |
| hang | 12 | 34.37 | 20.447.575 |
| other | 5 | 7.39 | 5.681.107 |
| pass | 24 | 49.41 | 31.067.078 |

### Reading of material

| Category | Calls | Distinct | Re-reads | Share of all tool calls |
|---|---|---|---|---|
| hal_docs | 39 | 21 | 18 | 0.097 |
| raw_ref | 0 | 0 | 0 | 0.0 |
| spec | 16 | 16 | 0 | 0.04 |
| tests | 10 | 9 | 1 | 0.025 |
| own_code | 15 | 5 | 10 | 0.037 |

_Iteration = a run of the agent's (`*_hal`) test, including a compile failure. Full pass = the same selection of variants as the original (same -k, the whole test file, or the same number of selected and deselected variants), pass count >= original, 0 failed. Input tokens up to the first pass are the sum of `usage` from the transcript messages up to that moment; output tokens cannot be counted reliably per moment from the transcript, so they are given only in total (from the `result` event)._
