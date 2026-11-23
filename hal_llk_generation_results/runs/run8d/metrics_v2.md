_metrics.py version 2_

| Metric | kernel_run8d |
|---|---|
| Duration (min) | 118.5 |
|   of which model time (min) | 59.2 |
| Number of steps (turns) | 279 |
| Input tokens (uncached) | 476 |
| Input tokens written to cache | 1.436.182 |
| Input tokens read from cache | 88.278.092 |
| Output tokens | 373.399 |
| Cache share of input | 0.984 |
| Cost (USD) | 34.48 |
| Bash commands with pytest | 82 |
| Pytest calls in them | 85 |
|   on the card | 67 |
|   compile only | 0 |
|   compile error | 10 |
|   wrong results | 21 |
|   hang / timeout | 8 |
|   pass | 20 |
|   denied | 3 |
| Iterations (agent test) | 74 |
| First pass of any selection (command no.) | 3 |
| Card resets (tt-smi -r) | 14 |
| File edits (Edit / Write) | 85 / 27 |
| File reads (Read) | 106 |
| Helper agents | 8 |
| Confluence / MCP calls | 0 |
| Denied tool calls | 33 |

### Flow of pytest invocations

| # | Call | Outcome | Passed | Failed | Selection | Command |
|---|---|---|---|---|---|---|
| 1 | 1 | collect | — | — | full | `cd /proj_sw/user_dev/ctr-sbratic/results/hal-runs/20261122_005259_B_c0e38f1f_claude-opus-5-5/ws/tt_metal/tt-llk/tests && timeout 300 pytest python_tests/test_el` |
| 2 | 1 | empty | 0 | 0 | subset:-k | `timeout 900 pytest python_tests/test_eltwise_binary.py -k 'test_eltwise_binary_int8_format and add and 512 and 32, 32' -q 2>&1 \| tail -15` |
| 3 | 1 | pass | 16 | 0 | subset:-k | `time timeout 900 pytest python_tests/test_eltwise_binary.py -k 'test_eltwise_binary_int8_format and add' -q 2>&1 \| tail -15` |
| 4 | 1 | denied | — | — | full | `timeout 300 pytest python_tests/test_eltwise_binary_hal.py -k 'add' --collect-only -q 2>&1 \| grep -E "test_eltwise_binary\[dest_acc:No-formats:Float16_b->Float1` |
| 5 | 1 | collect | — | — | subset:-k | `timeout 300 pytest python_tests/test_eltwise_binary_hal.py -k 'test_eltwise_binary and not bfp4 and not reuse and not int8 and Elwadd and None_ and transpose_sr` |
| 6 | 1 | compile_error | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:None_-math_op:Elwadd-math_fid` |
| 7 | 1 | denied | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:None_-math_op:Elwadd-math_fid` |
| 8 | 1 | compile_error | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:None_-math_op:Elwadd-math_fid` |
| 9 | 1 | compile_error | 0 | 1 | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:None_-math_op:Elwadd-math_fid` |
| 10 | 1 | compile_error | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:None_-math_op:Elwadd-math_fid` |
| 11 | 1 | compile_error | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:None_-math_op:Elwadd-math_fid` |
| 12 | 1 | compile_error | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:None_-math_op:Elwadd-math_fid` |
| 13 | 1 | compile_error | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:None_-math_op:Elwadd-math_fid` |
| 14 | 1 | hang | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:None_-math_op:Elwadd-math_fid` |
| 15 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r 2>&1 \| tail -3; timeout 600 pytest "python_tests/test_eltwise_binary.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:None_-m` |
| 16 | 1 | hang | 0 | 1 | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:None_-math_op:Elwadd-math_fid` |
| 17 | 1 | pass | 1 | 0 | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:Non` |
| 18 | 1 | compile_error | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:None_-math_op:Elwadd-math_fid` |
| 19 | 1 | hang | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:None_-math_op:Elwadd-math_fid` |
| 20 | 1 | hang | — | — | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:Non` |
| 21 | 1 | hang | — | — | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:Non` |
| 22 | 1 | hang | — | — | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:Non` |
| 23 | 1 | other | — | — | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:Non` |
| 24 | 1 | hang | — | — | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:Non` |
| 25 | 1 | hang | — | — | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:Non` |
| 26 | 1 | other | — | — | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:Non` |
| 27 | 1 | other | — | — | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:Non` |
| 28 | 1 | other | — | — | subset:nodeids | `tt-smi -r 2>&1 \| tail -1; timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:Non` |
| 29 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:None_-math_op:Elwadd-math_fid` |
| 30 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:None_-math_op:Elwadd-math_fid` |
| 31 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:None_-math_op:Elwadd-math_fid` |
| 32 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:None_-math_op:Elwadd-math_fid` |
| 33 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Bfp8_b->Bfp8_b-broadcast_type:None_-math_op:Elwadd-math_fid` |
| 34 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 35 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 36 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 37 | 1 | denied | — | — | subset:nodeids | `PYTHONPATH=hal_llk timeout 600 pytest -p dbg_plugin "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broad` |
| 38 | 1 | fail | 0 | 2 | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 39 | 1 | fail | 0 | 2 | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 40 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 41 | 1 | fail | — | 1 | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 42 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 43 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 44 | 1 | compile_error | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 45 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 46 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 47 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 48 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 49 | 1 | compile_error | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 50 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 51 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 52 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 53 | 1 | fail | 0 | 2 | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 54 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 55 | 1 | fail | 0 | 1 | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 56 | 1 | fail | 0 | 2 | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 57 | 1 | fail | 0 | 2 | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 58 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 59 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 60 | 1 | other | — | — | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 61 | 1 | pass | 1 | 0 | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 62 | 1 | pass | 2 | 0 | subset:nodeids | `timeout 600 pytest "python_tests/test_eltwise_binary_hal.py::test_eltwise_binary[dest_acc:No-formats:Float16_b->Float16_b-broadcast_type:None_-math_op:Elwadd-ma` |
| 63 | 1 | fail | 372 | 24 | subset:-k | `timeout 3000 pytest python_tests/test_eltwise_binary_hal.py -k 'test_eltwise_binary and not bfp4 and not reuse and not int8 and add' -q -n 8 2>&1 \| tail -5` |
| 64 | 1 | fail | 372 | 24 | subset:-k | `timeout 3000 pytest python_tests/test_eltwise_binary_hal.py -k 'test_eltwise_binary and not bfp4 and not reuse and not int8 and add' -q -n 8 --tb=no -rf 2>&1 \| ` |
| 65 | 1 | fail | 372 | 24 | subset:-k | `timeout 3000 pytest python_tests/test_eltwise_binary_hal.py -k 'test_eltwise_binary and not bfp4 and not reuse and not int8 and add' -q --tb=no -rf -p no:loggin` |
| 66 | 1 | empty | 0 | 0 | subset:-k | `timeout 3000 pytest python_tests/test_eltwise_binary_hal.py -k 'test_eltwise_binary and not bfp4 and not reuse and not int8 and add' -v --tb=no -p no:logging 2>` |
| 67 | 1 | pass | 36 | 0 | subset:-k | `timeout 3000 pytest python_tests/test_eltwise_binary_hal.py -k 'test_eltwise_binary and not bfp4 and not reuse and not int8 and add and transpose_srca:Yes' -q -` |
| 68 | 1 | other | — | — | subset:-k | `timeout 3000 pytest python_tests/test_eltwise_binary_hal.py -k '(bfp4 or int8) and add' -v --tb=no -p no:logging 2>&1 \| rev \| sort \| rev \| cut -c40-250 \| tail -` |
| 69 | 1 | empty | 0 | 0 | subset:-k | `timeout 3000 pytest python_tests/test_eltwise_binary_hal.py -k '(bfp4 or int8 or reuse) and add' -v --tb=no -p no:logging 2>&1 \| rev \| sort \| rev \| cut -c40-250` |
| 70 | 1 | pass | 1.240 | 0 | subset:-k | `timeout 3000 pytest python_tests/test_eltwise_binary_hal.py -k '(bfp4 or int8) and add' -q --tb=no -p no:logging 2>&1 \| tail -1; timeout 3000 pytest python_test` |
| 70 | 2 | pass | 12 | 0 | subset:-k | `timeout 3000 pytest python_tests/test_eltwise_binary_hal.py -k '(bfp4 or int8) and add' -q --tb=no -p no:logging 2>&1 \| tail -1; timeout 3000 pytest python_test` |
| 70 | 3 | fail | 0 | 12 | subset:-k | `timeout 3000 pytest python_tests/test_eltwise_binary_hal.py -k '(bfp4 or int8) and add' -q --tb=no -p no:logging 2>&1 \| tail -1; timeout 3000 pytest python_test` |
| 71 | 1 | pass | 12 | 0 | subset:-k | `timeout 3000 pytest python_tests/test_eltwise_binary_hal.py -k 'reuse and add and DEST_TO_SRCA' -q --tb=no -p no:logging 2>&1 \| tail -1` |
| 72 | 1 | pass | 1.660 | 0 | full | `timeout 3600 pytest python_tests/test_eltwise_binary_hal.py -k 'add' -q --tb=no -p no:logging -n 8 2>&1 \| tail -1 \| cut -c1-200` |
| 73 | 1 | pass | 1.089 | 0 | subset:-k | `timeout 3600 pytest python_tests/test_eltwise_binary_hal.py -k 'test_eltwise_binary and not bfp4 and not int8 and not reuse and (Elwsub or Elwmul)' -q --tb=no -` |
| 74 | 1 | fail | 2.716 | 32 | subset:-k | `timeout 3600 pytest python_tests/test_eltwise_binary_hal.py -k 'not add' -q --tb=no -p no:logging -n 8 2>&1 \| tail -1 \| cut -c1-200` |
| 75 | 1 | fail | 0 | 32 | subset:-k | `timeout 3600 pytest python_tests/test_eltwise_binary_hal.py -k 'not add and reuse and Elwmul' -q --tb=no -p no:logging -n 8 2>&1 \| tail -1 \| cut -c1-200; timeou` |
| 75 | 2 | pass | 24 | 0 | subset:-k | `timeout 3600 pytest python_tests/test_eltwise_binary_hal.py -k 'not add and reuse and Elwmul' -q --tb=no -p no:logging -n 8 2>&1 \| tail -1 \| cut -c1-200; timeou` |
| 76 | 1 | fail | 0 | 32 | subset:-k | `timeout 3600 pytest python_tests/test_eltwise_binary_hal.py -k 'not add and reuse and Elwmul' -q --tb=no -p no:logging -n 8 2>&1 \| tail -1 \| cut -c1-200` |
| 77 | 1 | pass | 1.660 | 0 | full | `timeout 3600 pytest python_tests/test_eltwise_binary_hal.py -k 'add' -q --tb=no -p no:logging -n 8 2>&1 \| tail -1 \| cut -c1-200` |
| 78 | 1 | pass | 1.660 | 0 | full | `tt-smi -r 2>&1 \| tail -1; time timeout 590 pytest python_tests/test_eltwise_binary_hal.py -k 'add' -q --tb=no -p no:logging 2>&1 \| tail -1 \| cut -c1-200` |
| 79 | 1 | pass | 1.660 | 0 | full | `tt-smi -r 2>&1 \| tail -1; timeout 590 pytest python_tests/test_eltwise_binary_hal.py -k 'add' 2>&1 \| tail -1` |
| 80 | 1 | pass | 1.660 | 0 | full | `timeout 590 pytest python_tests/test_eltwise_binary.py -k 'add' 2>&1 \| tail -1` |
| 81 | 1 | pass | 1.660 | 0 | full | `timeout 590 pytest python_tests/test_eltwise_binary_hal.py -k 'add' 2>&1 \| tail -1` |
| 82 | 1 | pass | 1.660 | 0 | full | `tt-smi -r 2>&1 \| tail -1; timeout 590 pytest python_tests/test_eltwise_binary_hal.py -k 'add' 2>&1 \| tail -1` |

_The outcome classification is heuristic (based on the pytest output) and is meant as a guide._

### First full pass per test

| Test | Iteration (total / of that test) | Command no. | Time (min) | Input tokens | Original passed |
|---|---|---|---|---|---|
| test_eltwise_binary_hal | 64 / 64 | 72 | 89.59 | 78.736.038 | 1.660 |

### First full pass of the final code (after the last change of the agent's code)

| Test | Iteration (total / of that test) | Command no. | Time (min) | Input tokens |
|---|---|---|---|---|
| test_eltwise_binary_hal | 70 / 70 | 77 | 99.96 | 83.834.480 |

Iterations total: 74; all tests fully passed at iteration: 64; changes of the agent's code: 102; ambiguous commands (the number of pytest calls and final lines does not match): 1

| Test | Success@5 | Success@10 | Success@15 | Success@20 | Success@25 | Success@50 |
|---|---|---|---|---|---|---|
| test_eltwise_binary_hal | no | no | no | no | no | no |

### Cost by iteration outcome kind

| Outcome | Iterations | Minutes | Input tokens |
|---|---|---|---|
| compile_error | 10 | 29.55 | 31.014.304 |
| fail | 21 | 24.1 | 16.532.819 |
| hang | 8 | 10.37 | 7.775.387 |
| other | 18 | 15.65 | 12.341.669 |
| pass | 17 | 37.21 | 19.843.146 |

### Reading of material

| Category | Calls | Distinct | Re-reads | Share of all tool calls |
|---|---|---|---|---|
| hal_docs | 40 | 18 | 22 | 0.112 |
| raw_ref | 0 | 0 | 0 | 0.0 |
| spec | 19 | 19 | 0 | 0.053 |
| tests | 31 | 23 | 8 | 0.087 |
| own_code | 23 | 10 | 13 | 0.065 |

_Iteration = a run of the agent's (`*_hal`) test, including a compile failure. Full pass = the same selection of variants as the original (same -k, the whole test file, or the same number of selected and deselected variants), pass count >= original, 0 failed. Input tokens up to the first pass are the sum of `usage` from the transcript messages up to that moment; output tokens cannot be counted reliably per moment from the transcript, so they are given only in total (from the `result` event)._
