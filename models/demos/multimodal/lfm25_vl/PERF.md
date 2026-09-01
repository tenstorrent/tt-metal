# Model performance and accuracy

Performance collected from [demo/vision_demo.py](demo/vision_demo.py) (`batch1-notrace` on Koyeb N300, real HF weights).

Note: this is a functional bring-up of LFM2.5-VL-1.6B. ShortConv decode state is host-side, so decode throughput is not yet optimized / traced. Numbers below are from the first N300 hardware run (no-trace).

## Performance

| Model          | Device | Speed (t/s/u) | TTFT (ms) |
|----------------|--------|---------------|-----------|
| LFM2.5-VL-1.6B | N150   | TBD           | TBD       |
| LFM2.5-VL-1.6B | N300   | TBD\*         | 711.21    |

\*Decode tokens/sec not reported yet: ShortConv decode uses a host-resident state path (`enable_trace=False`), so a fair t/s/u number needs a dedicated timed decode loop after further optimization.

## Accuracy

Top-1 / Top-5 image-classification style metrics are N/A for this OCR / VLM bring-up. Unit-test PCC vs torch reference on N300:

| Module     | Device | PCC (measured) | Status |
|------------|--------|----------------|--------|
| ShortConv  | N300   | ~0.9998        | PASSED |
| MLP        | N300   | ~0.9996        | PASSED |
| Projector  | N300   | ~0.9999        | PASSED |

| Model          | Device | Top-1 (%) | Top-5 (%) |
|----------------|--------|-----------|-----------|
| LFM2.5-VL-1.6B | N150   | TBD       | TBD       |
| LFM2.5-VL-1.6B | N300   | TBD       | TBD       |
