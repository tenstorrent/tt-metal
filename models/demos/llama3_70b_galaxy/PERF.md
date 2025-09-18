# Llama-3.3-70B on Galaxy Performance

This file contains the measured performance of Llama-3.3-70B on Galaxy systems, coupled with the run commands we have used to generate these numbers.

Please note that using more recent versions of the software stack (TT-Metal and vLLM) might lead to different performance numbers.


## 25 July 2025

- TT-Metal: [633160e](https://github.com/tenstorrent/tt-metal/commit/633160efc176b71dc35b6d15be0e9421b986493d)
- vLLM: [0c0ee9e](https://github.com/tenstorrent/vllm/commit/0c0ee9e6d81bbb26d44229992ad2273e8ea1052b)
- TT-Inference-server: [3380ea0](https://github.com/tenstorrent/tt-inference-server/commit/3380ea073e53f20ea782626f42c406e5cf4260c8)


### TT-Metal runs

```
pytest models/demos/llama3_70b_galaxy/demo/text_demo.py -k performance-batch-1
```
To run different sequence lengths you can use the following configurations:
- performance-long-4k-b1
- performance-long-8k-b1
- performance-long-32k-b1
- performance-long-64k-b1
- performance-long-128k-b1

All of the above will run for batch 1. To run a different batch size just add the override flag to the run command `--batch_size <NUMBER>`.

Similarly, if you want to run a different input prompt file with different length you can use the override `--input_prompts “models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_1k.json”` (this example will run the 1K prompts instead).

| Input length | Output length | Batch  | TTFT (per user)   | Token/s/u (avg of all decoded tokens)                                        |
|--------------|---------------|--------|-------------------|------------------------------------------------------------------------------|
| 128          | 128           | 1, 32  | 59.64 ms          | 14.01 ms, 71.38 t/s/u (measured at 128th token=117 prefill + 11 decode) <br> 14.28 ms, 70.01 t/s/u (avg) |
| 1K           | 128           | 1      | 160.86 ms         | 16.53 ms, 60.48 t/s/u                                                    |
| 1K           | 128           | 32     | 161.66 ms         | 17.58 ms, 56.89 t/s/u                                                     |
| 4K           | 128           | 1      | 625.08 ms         | 18.5 ms, 54.07 t/s/u                                                      |
| 4K           | 128           | 32     | 626.85 ms         | 22.43 ms, 44.58 t/s/u                                                     |
| 8K           | 128           | 1      | 1,382.3 ms        | 21.41 ms, 46.72 t/s/u                                                     |
| 16K          | 128           | 1      | 3,403.12 ms       | 27.49 ms, 36.38 t/s/u                                                     |
| 32K          | 128           | 1      | 9,482.38 ms       | 38.17 ms, 26.2 t/s/u                                                      |
| 64K          | 128           | 1      | 29,697.47 ms      | 60.87 ms, 16.43 t/s/u                                                     |
| 128K         | 128           | 1      | 102,310.21 ms     | 91.25 ms, 10.96 t/s/u                                                     |

### vLLM Runs

For vLLM runs, you'll need to install [Tenstorrent's vLLM fork](https://github.com/tenstorrent/vllm/blob/dev/tt_metal/README.md).

vLLM offline run command:
```
VLLM_RPC_TIMEOUT=100000 TT_LLAMA_TEXT_VER=llama3_70b_galaxy python examples/offline_inference_tt.py  --model meta-llama/Llama-3.3-70B-Instruct --override_tt_config '{"dispatch_core_axis": "col", "sample_on_device_mode": "all", "fabric_config": "FABRIC_1D_RING", "worker_l1_size": 1344544, "trace_region_size": 95693824}' --greedy_sampling --num_repeat_prompts 2 --num_scheduler_steps 30 --async_engine --measure_perf
```

vLLM server run command:
```
TT_LLAMA_TEXT_VER=llama3_70b_galaxy VLLM_RPC_TIMEOUT=900000 python examples/server_example_tt.py --model "meta-llama/Llama-3.3-70B-Instruct" --override_tt_config '{"dispatch_core_axis": "col", "sample_on_device_mode": "all", "fabric_config": "FABRIC_1D_RING", "worker_l1_size": 1344544, "trace_region_size": 95693824}' --num_scheduler_steps 30
```

To send requests to vLLM the server, you will need [TT-Inference-Server](https://github.com/tenstorrent/tt-inference-server/tree/dev).
```
export HF_MODEL_REPO_ID='meta-llama/Llama-3.3-70B-Instruct'

cd tt-inference-server/vllm-tt-metal-llama3/src
python example_requests_client.py --num_concurrent 32 --prompt_json_path "vllm_server_prompts.json"
```

**Note:** Please send the request twice, to warmup the cache and get accurate performance numbers.
**Note:** The file containing server prompts can be found in tt-metal at `models/demos/llama3_70b_galaxy/demo/sample_prompts/vllm_server_prompts.json `.

| Command      | Input length | Output length | Batch  | TTFT (per user)   | Token/s/u (avg of all decoded tokens) |
|--------------|--------------|---------------|--------|-------------------|---------------------------------------|
| vLLM Offline | 128          | 128           | 32     | 69.14 ms          | 14.92 ms, 67.03 t/s/u                 |
| vLLM server  | 128          | 128           | 32     | 74.91 ms          | 15.12 ms, 66.15 t/s/u                 |


To Run the TT-Inference-Server release tests (Benchmark and eval), you can run:
```
python run.py --model Llama-3.3-70B-Instruct --device galaxy --workflow release
```

Please refer to the [TT-Inference-Server](https://github.com/tenstorrent/tt-inference-server/tree/dev) repo for more information.
