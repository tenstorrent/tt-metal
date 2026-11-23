_metrics.py version 2_

| Metric | kernel_run8e |
|---|---|
| Duration (min) | 112.9 |
|   of which model time (min) | 51.5 |
| Number of steps (turns) | 317 |
| Input tokens (uncached) | 540 |
| Input tokens written to cache | 1.537.383 |
| Input tokens read from cache | 98.212.541 |
| Output tokens | 338.806 |
| Cache share of input | 0.985 |
| Cost (USD) | 34.61 |
| Bash commands with pytest | 78 |
| Pytest calls in them | 83 |
|   on the card | 63 |
|   compile only | 5 |
|   compile error | 1 |
|   wrong results | 25 |
|   hang / timeout | 2 |
|   pass | 20 |
|   denied | 7 |
| Iterations (agent test) | 62 |
| First pass of any selection (command no.) | 2 |
| Card resets (tt-smi -r) | 21 |
| File edits (Edit / Write) | 106 / 17 |
| File reads (Read) | 133 |
| Helper agents | 8 |
| Confluence / MCP calls | 0 |
| Denied tool calls | 39 |

### Flow of pytest invocations

| # | Call | Outcome | Passed | Failed | Selection | Command |
|---|---|---|---|---|---|---|
| 1 | 1 | collect | — | — | full | `cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261122_005334_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 600 pytest python_tests/test_el` |
| 2 | 1 | pass | 16 | 0 | subset:-k | `time timeout 900 pytest python_tests/test_eltwise_binary.py -k 'add and int8' -q -x 2>&1 \| tail -15` |
| 3 | 1 | denied | — | — | subset:-k | `timeout 600 pytest python_tests/test_eltwise_binary_hal.py -k 'add and int8 and 32, 32]-tile_dimensions:[32, 32]' -x -q 2>&1 \| grep -E "error\|Error\|PASS\|FAIL\|pa` |
| 4 | 1 | denied | — | — | subset:-k | `timeout 600 pytest python_tests/test_eltwise_binary_hal.py -k 'add and int8 and 32, 32]-tile_dimensions:[32, 32]' -x -q > /proj_sw/user_dev/ctr-sbratic/results/` |
| 5 | 1 | error | — | — | subset:-k | `timeout 600 pytest python_tests/test_eltwise_binary_hal.py -k 'add and int8 and 32, 32]-tile_dimensions:[32, 32]' -x -q > /proj_sw/user_dev/ctr-sbratic/results/` |
| 6 | 1 | collect | — | — | full | `timeout 600 pytest python_tests/test_eltwise_binary_hal.py -k add --collect-only -q > /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261122_005334_B_c0e38f1f_` |
| 7 | 1 | error | — | — | subset:nodeids | `timeout 900 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 8 | 1 | error | — | — | subset:nodeids | `timeout 900 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 9 | 1 | error | — | — | subset:nodeids | `timeout 900 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 10 | 1 | error | — | — | subset:nodeids | `timeout 900 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 11 | 1 | error | — | — | subset:nodeids | `timeout 900 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 12 | 1 | error | — | — | subset:nodeids | `timeout 900 pytest --logging-level=debug "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:N` |
| 13 | 1 | error | — | — | subset:nodeids | `tt-smi -r > /dev/null 2>&1; timeout 900 pytest --logging-level=debug "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b` |
| 14 | 1 | error | — | — | subset:nodeids | `tt-smi -r > /dev/null 2>&1; timeout 900 pytest --logging-level=debug "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b` |
| 15 | 1 | error | — | — | subset:nodeids | `tt-smi -r > /dev/null 2>&1; timeout 900 pytest --logging-level=debug "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b` |
| 16 | 1 | error | — | — | subset:nodeids | `tt-smi -r > /dev/null 2>&1; timeout 900 pytest --logging-level=debug "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b` |
| 17 | 1 | error | — | — | subset:nodeids | `tt-smi -r > /dev/null 2>&1; timeout 900 pytest --logging-level=debug "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b` |
| 18 | 1 | error | — | — | subset:nodeids | `tt-smi -r > /dev/null 2>&1; timeout 900 pytest --logging-level=debug "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b` |
| 19 | 1 | error | — | — | subset:nodeids | `tt-smi -r > /dev/null 2>&1; timeout 900 pytest --logging-level=debug "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b` |
| 20 | 1 | error | — | — | subset:nodeids | `tt-smi -r > /dev/null 2>&1; timeout 900 pytest --logging-level=debug "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b` |
| 21 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > /dev/null 2>&1; timeout 900 pytest --logging-level=debug "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b` |
| 22 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r > /dev/null 2>&1; timeout 900 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_` |
| 23 | 1 | denied | — | — | subset:nodeids | `tt-smi -r > /dev/null 2>&1; timeout 900 pytest --logging-level=debug "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b` |
| 24 | 1 | error | — | — | subset:nodeids | `tt-smi -r > /dev/null 2>&1; timeout 900 pytest --logging-level=debug "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b` |
| 25 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > /dev/null 2>&1; timeout 900 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_` |
| 26 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > /dev/null 2>&1; timeout 900 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_` |
| 27 | 1 | hang | 0 | 1 | subset:nodeids | `tt-smi -r > /dev/null 2>&1; timeout 900 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_` |
| 28 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > /dev/null 2>&1; timeout 900 pytest --logging-level=debug "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b` |
| 29 | 1 | fail | 0 | 1 | subset:nodeids | `tt-smi -r > /dev/null 2>&1; timeout 900 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_` |
| 30 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 900 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 31 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 900 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 32 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 900 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 33 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 900 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 34 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 900 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 35 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 900 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 36 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 900 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 37 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 900 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 38 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 900 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 39 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 900 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 40 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 900 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 41 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 900 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 42 | 1 | compile_only | 1.660 | 0 | full | `nproc && timeout 3000 pytest --compile-producer -n 48 python_tests/test_eltwise_binary_hal.py -k add -q > /proj_sw/user_dev/ctr-sbratic/results/hal-runs/2026112` |
| 43 | 1 | fail | 16 | 24 | subset:-k | `timeout 580 pytest --compile-consumer python_tests/test_eltwise_binary_hal.py -k "add and (int8 or dest_reuse)" -q > /proj_sw/user_dev/ctr-sbratic/results/hal-r` |
| 44 | 1 | fail | 16 | 24 | subset:-k | `timeout 580 pytest --compile-consumer python_tests/test_eltwise_binary_hal.py -k "add and (int8 or dest_reuse)" -q -rA 2>&1 \| tail -45 \| cut -c1-260` |
| 45 | 1 | fail | 16 | 8 | subset:-k | `timeout 580 pytest --compile-consumer python_tests/test_eltwise_binary_hal.py -k "add and dest_reuse" -q -rf --tb=no 2>&1 \| tail -12 \| cut -c1-260` |
| 46 | 1 | fail | 16 | 8 | subset:-k | `timeout 580 pytest --compile-consumer python_tests/test_eltwise_binary_hal.py -k "add and dest_reuse" -q -rf --tb=no 2>/dev/null \| tail -10 \| cut -c1-300` |
| 47 | 1 | denied | — | — | subset:-k | `for f in "Float16_b" "Float32" "Bfp8_b"; do for r in SRCA SRCB; do echo "$f $r: $(timeout 300 pytest --compile-consumer python_tests/test_eltwise_binary_hal.py ` |
| 48 | 1 | pass | 8 | 0 | subset:-k | `timeout 300 pytest --compile-consumer python_tests/test_eltwise_binary_hal.py -k "add and dest_reuse and Float16_b" -q --tb=no 2>/dev/null \| tail -1; timeout 30` |
| 48 | 2 | pass | 8 | 0 | subset:-k | `timeout 300 pytest --compile-consumer python_tests/test_eltwise_binary_hal.py -k "add and dest_reuse and Float16_b" -q --tb=no 2>/dev/null \| tail -1; timeout 30` |
| 48 | 3 | fail | 0 | 8 | subset:-k | `timeout 300 pytest --compile-consumer python_tests/test_eltwise_binary_hal.py -k "add and dest_reuse and Float16_b" -q --tb=no 2>/dev/null \| tail -1; timeout 30` |
| 48 | 4 | fail | 8 | 4 | subset:-k | `timeout 300 pytest --compile-consumer python_tests/test_eltwise_binary_hal.py -k "add and dest_reuse and Float16_b" -q --tb=no 2>/dev/null \| tail -1; timeout 30` |
| 49 | 1 | fail | 16 | 8 | subset:-k | `timeout 580 pytest -n 8 python_tests/test_eltwise_binary_hal.py -k "add and dest_reuse" -q --tb=line 2>/dev/null \| tail -3 \| cut -c1-300` |
| 50 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 580 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_dest_reuse[reuse_dest_type:DEST_TO_SRCA-math_op:Elwadd-formats:Bfp8_b->Bfp8_b-m` |
| 51 | 1 | compile_only | 1.660 | 0 | full | `timeout 590 pytest --compile-producer -n 24 python_tests/test_eltwise_binary_hal.py -k add -q 2>/dev/null \| tail -1; timeout 400 pytest --compile-consumer pytho` |
| 51 | 2 | fail | 515 | 709 | subset:-k | `timeout 590 pytest --compile-producer -n 24 python_tests/test_eltwise_binary_hal.py -k add -q 2>/dev/null \| tail -1; timeout 400 pytest --compile-consumer pytho` |
| 52 | 1 | denied | — | — | subset:-k | `PYTHONPATH=/proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261122_005334_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests/hal_llk timeout 590 pytest -p fail` |
| 53 | 1 | fail | 732 | 888 | subset:-k | `timeout 590 pytest --compile-consumer python_tests/test_eltwise_binary_hal.py -k "add and not dest_reuse and not int8" -q --tb=no 2>/dev/null \| tail -1; ls -la ` |
| 54 | 1 | denied | — | — | subset:--lf | `for k in "None_ and transpose_srca:No" "None_ and transpose_srca:Yes" "Row" "Column" "Scalar" "dest_acc:No" "dest_acc:Yes" "Float16_b->Float16_b" "Bfp8_b->Bfp8_` |
| 55 | 1 | collect | — | — | subset:--lf | `pytest --lf --co -q python_tests/test_eltwise_binary_hal.py -k "add and not dest_reuse and not int8 and None_ and transpose_srca:No" 2>/dev/null \| tail -1; pyte` |
| 56 | 1 | collect | — | — | subset:--lf | `pytest --lf --co -q python_tests/test_eltwise_binary_hal.py -k "add and not dest_reuse and not int8 and None_ and transpose_srca:No" 2>/dev/null \| cut -d'[' -f2` |
| 57 | 1 | collect | — | — | subset:--lf | `pytest --lf --co -q python_tests/test_eltwise_binary_hal.py -k "add and not dest_reuse and not int8" 2>/dev/null \| cut -d'[' -f2 \| cut -d'-' -f1-3 \| sort \| uniq` |
| 58 | 1 | collect | — | — | subset:--lf | `pytest --lf --co -q python_tests/test_eltwise_binary_hal.py -k "add and not dest_reuse and not int8 and Bfp8_b-Float16_b" 2>/dev/null \| cut -d'[' -f2- \| cut -d'` |
| 59 | 1 | collect | — | — | subset:--lf | `pytest --lf --co -q python_tests/test_eltwise_binary_hal.py -k "add and not dest_reuse and not int8 and formats:Bfp8_b and Float16_b-broadcast" 2>/dev/null \| cu` |
| 60 | 1 | compile_only | 1.660 | 0 | full | `timeout 590 pytest --compile-producer -n 24 python_tests/test_eltwise_binary_hal.py -k add -q -p no:cacheprovider 2>/dev/null \| tail -1` |
| 61 | 1 | pass | 888 | 0 | subset:--lf | `timeout 590 pytest --compile-consumer --lf python_tests/test_eltwise_binary_hal.py -k "add and not dest_reuse and not int8" -q --tb=no 2>/dev/null \| tail -1` |
| 62 | 1 | fail | 24 | 16 | subset:-k | `timeout 590 pytest --compile-consumer python_tests/test_eltwise_binary_hal.py -k "add and (dest_reuse or int8)" -q --tb=no 2>/dev/null \| tail -1` |
| 63 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 590 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_type:None_-math_fidelity:` |
| 64 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 590 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_type:None_-math_fidelity:` |
| 65 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 590 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_type:None_-math_fidelity:` |
| 66 | 1 | compile_error | 0 | 1 | subset:nodeids | `timeout 590 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_type:None_-math_fidelity:` |
| 67 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 590 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_type:None_-math_fidelity:` |
| 68 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 590 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_type:None_-math_fidelity:` |
| 69 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 590 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary_int8_format[dest_acc:Yes-formats:Int8->Int8-broadcast_type:None_-math_fidelity:` |
| 70 | 1 | compile_only | 1.660 | 0 | full | `timeout 590 pytest --compile-producer -n 24 python_tests/test_eltwise_binary_hal.py -k add -q -p no:cacheprovider 2>/dev/null \| tail -1; timeout 300 pytest --co` |
| 70 | 2 | pass | 40 | 0 | subset:-k | `timeout 590 pytest --compile-producer -n 24 python_tests/test_eltwise_binary_hal.py -k add -q -p no:cacheprovider 2>/dev/null \| tail -1; timeout 300 pytest --co` |
| 71 | 1 | pass | 1.620 | 0 | subset:-k | `timeout 590 pytest --compile-consumer python_tests/test_eltwise_binary_hal.py -k "add and not dest_reuse and not int8" -q --tb=no 2>/dev/null \| tail -1` |
| 72 | 1 | compile_only | 3.320 | 0 | subset:-k | `timeout 590 pytest --compile-producer -n 24 python_tests/test_eltwise_binary_hal.py python_tests/test_eltwise_binary.py -k add -q -p no:cacheprovider 2>/dev/nul` |
| 73 | 1 | denied | — | — | full | `tt-smi -r > /dev/null 2>&1; echo "reset rc=$?"; timeout 590 pytest --compile-consumer python_tests/test_eltwise_binary_hal.py -k 'add' -q --tb=no 2>/dev/null \| ` |
| 74 | 1 | pass | 1.660 | 0 | full | `tt-smi -r > /dev/null 2>&1; timeout 590 pytest --compile-consumer python_tests/test_eltwise_binary_hal.py -k 'add' -q --tb=no 2>/dev/null \| tail -1` |
| 75 | 1 | pass | 1.660 | 0 | full | `timeout 590 pytest --compile-consumer python_tests/test_eltwise_binary.py -k 'add' -q --tb=no 2>/dev/null \| tail -1` |
| 76 | 1 | pass | 1.660 | 0 | full | `timeout 590 pytest --compile-consumer python_tests/test_eltwise_binary_hal.py -k 'add' -q --tb=no 2>/dev/null \| tail -1` |
| 77 | 1 | pass | 1.660 | 0 | full | `tt-smi -r > /dev/null 2>&1; timeout 590 pytest --compile-consumer python_tests/test_eltwise_binary_hal.py -k 'add' -q --tb=no 2>/dev/null \| tail -1` |
| 78 | 1 | pass | 40 | 0 | subset:-k | `timeout 590 pytest python_tests/test_eltwise_binary_hal.py -k 'add and (dest_reuse or int8)' -q --tb=no 2>/dev/null \| tail -1` |

_The outcome classification is heuristic (based on the pytest output) and is meant as a guide._

### First full pass per test

| Test | Iteration (total / of that test) | Command no. | Time (min) | Input tokens | Original passed |
|---|---|---|---|---|---|
| test_eltwise_binary_hal | 59 / 59 | 74 | 101.54 | 96.011.570 | 1.660 |

### First full pass of the final code (after the last change of the agent's code)

| Test | Iteration (total / of that test) | Command no. | Time (min) | Input tokens |
|---|---|---|---|---|
| test_eltwise_binary_hal | 59 / 59 | 74 | 101.54 | 96.011.570 |

Iterations total: 62; all tests fully passed at iteration: 59; changes of the agent's code: 113; ambiguous commands (the number of pytest calls and final lines does not match): 0

| Test | Success@5 | Success@10 | Success@15 | Success@20 | Success@25 | Success@50 |
|---|---|---|---|---|---|---|
| test_eltwise_binary_hal | no | no | no | no | no | no |

### Cost by iteration outcome kind

| Outcome | Iterations | Minutes | Input tokens |
|---|---|---|---|
| compile_error | 1 | 0.53 | 712.669 |
| error | 16 | 39.38 | 46.239.130 |
| fail | 25 | 30.87 | 21.821.913 |
| hang | 2 | 2.5 | 1.845.092 |
| pass | 18 | 38.24 | 28.382.715 |

### Reading of material

| Category | Calls | Distinct | Re-reads | Share of all tool calls |
|---|---|---|---|---|
| hal_docs | 50 | 20 | 30 | 0.123 |
| raw_ref | 0 | 0 | 0 | 0.0 |
| spec | 18 | 18 | 0 | 0.044 |
| tests | 42 | 26 | 16 | 0.103 |
| own_code | 7 | 4 | 3 | 0.017 |

_Iteration = a run of the agent's (`*_hal`) test, including a compile failure. Full pass = the same selection of variants as the original (same -k, the whole test file, or the same number of selected and deselected variants), pass count >= original, 0 failed. Input tokens up to the first pass are the sum of `usage` from the transcript messages up to that moment; output tokens cannot be counted reliably per moment from the transcript, so they are given only in total (from the `result` event)._
