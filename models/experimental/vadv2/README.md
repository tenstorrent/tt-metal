# Transformer Block Bring-Up - Status Report

## Current Progress

- **Transformer block** bring-up is **complete**.
- Achieved **PCC** of **0.99**.
- Issues have been created for the fallbacks.

---

##  Ongoing Work

- Bring-up of the **full model** is still in progress.
- Additional integration and validation steps are pending.
- **Please note:** intermediate and unused code should be cleaned up

---

## Model Weights

To run the Transformer block, you must first download the pretrained weights.

### 🔗 Download Instructions

1. Go to the following Google Drive link:
[https://drive.google.com/file/d/1uufTSZMv9xOanBQiWy4vDrGEvxFDi0wR/view?usp=sharing]
2. Download the `weights.zip` file.
3. Unzip it inside the project root directory(`models/experimental/vadv2/`)


## How to Run the Transformer Block

Use the following command to run the Transformer block test:

```
pytest tests/ttnn/integration_tests/vadv2/test_tt_transformer.py
```
