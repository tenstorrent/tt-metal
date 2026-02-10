# vLLM Stress Test Cheat Sheet

## Start the test in background

```bash
tmux new -d -s stress "source /opt/venv/bin/activate && cd /tt-metal && ./run_vllm_server_stress.sh"
```

## Check on it (attach to the terminal)

```bash
tmux attach -t stress
```

Then **Ctrl+B, D** to detach again without stopping it.

## Check progress without attaching

```bash
cat /tt-metal/vllm_stress_summary_*.log
```

## Stop the test

```bash
tmux kill-session -t stress
pkill -9 -f "server_example_tt|tt_core_launcher|prterun.*tmp_vllm"
sleep 2
tt-smi -r
```

## Clean up logs before a new run

```bash
rm -f /tt-metal/vllm_stress_*.log
```
