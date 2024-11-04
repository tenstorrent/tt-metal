# Performance Report Analysis Tool

This tool analyzes performance traces from Metal operations, providing insights into throughput, bottlenecks, and optimization opportunities.

## Generating Performance Traces

1. Build Metal with performance tracing enabled:
```bash
./build_metal -p
```

2. Run your test with the tracy module to capture traces:
```bash
python -m tracy -r -p -v -m pytest path/to/test.py
```
This generates a CSV file containing operation timing data.

## Using Tracy Signposts

Tracy signposts mark specific sections of code for analysis. Add signposts to your Python code:

```python
import tracy

# Mark different sections of your code
tracy.signpost("Compilation pass")
model(input_data)

tracy.signpost("Performance pass")
for _ in range(10):
    model(input_data)
```

The tool uses the last signpost by default, which is typically the most relevant section for a performance test(e.g., the final iteration after compilation / warmup).

Common signpost usage:
- `--signpost name`: Analyze ops after the specified signpost
- `--ignore-signposts`: Analyze the entire trace

## Filtering Operations

The output of the performance report is a table of operations. Each operation is assigned a unique ID starting from 1. You can re-run the tool with different IDs to focus on specific sections of the trace.

Use `--id-range` to analyze specific sections:
```bash
# Analyze ops 5 through 10
python perf_report.py trace.csv --id-range 5-10

# Analyze from op 31 onwards
python perf_report.py trace.csv --id-range 31-

# Analyze up to op 12
python perf_report.py trace.csv --id-range -12
```

This is particularly useful for:
- Isolating decode pass in prefill+decode LLM inference
- Analyzing single transformer layers without embeddings/projections
- Focusing on specific model components

## Output Options

- `--min-percentage value`: Hide ops below specified % of total time (default: 0.5)
- `--color/--no-color`: Force colored/plain output
- `--csv`: Output in CSV format for further analysis
- `--no-advice`: Show only performance table, skip optimization advice

## Examples

Typical use:

```bash
python perf_report.py trace.csv
```

Build a table of all ops with no advice:

```bash
python perf_report.py trace.csv --no-advice
```

View ops 100-200 with advice:

```bash
python perf_report.py trace.csv --id-range 100-200
```

Export the table of ops and columns as a CSV file:

```bash
python perf_report.py trace.csv --csv my_report.csv
```

