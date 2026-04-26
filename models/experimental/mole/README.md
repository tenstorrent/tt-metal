# MoLE -- Mixture of Linear Experts on Tenstorrent (TTNN)

Time-series forecasting with Mixture-of-Linear-Experts, accelerated on Tenstorrent Wormhole hardware using TTNN.

MoLE Paper: https://arxiv.org/abs/2312.06786
Reference implementation: https://github.com/RogerNi/MoLE/


### 1. Smoke-test all bundled checkpoints

Downloads four example models plus the ETTh1 dataset and runs them through a quick inference test to verify that everything is working:

/root/tt-metal/models/experimental/mole/demo/run_all_demo_checkpoints.sh

### 2. Benchmark latency and throughput

Measures TTNN inference latency (ms) and throughput (sequences/sec). Expert overhead is calculated by comparing to a single-expert counterpart.

python3 -m models.experimental.mole.demo.benchmark \
  --base-model-type dlinear \
  --dataset-dir /root/tt-metal/models/experimental/mole/demo_checkpoints \
  --dataset-file ETTh1.csv \
  --seq-len 336 --pred-len 96 --num-experts 4 \
  --checkpoint-dir /root/tt-metal/models/experimental/mole/demo_checkpoints/etth1_DLinear_mole_sl336_pl96_td4_lr0.01_hd0.2_sd2021_MoLE_DLinear_ETTh1_ftM_sl336_ll336_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_DemoMatrix_0_32_4_f_mask_0.5_0.01_sd2021_hd0.2 \
  --checkpoint-file checkpoint.pth \
  --batch-size 8 \
  --warmup-iterations 100 \
  --measure-iterations 1000


**DLinear**

python3 -m models.experimental.mole.demo.benchmark \
  --base-model-type dlinear \
  --dataset-dir /root/tt-metal/models/experimental/mole/demo_checkpoints \
  --dataset-file ETTh1.csv \
  --seq-len 336 --pred-len 96 --num-experts 1 \
  --checkpoint-dir /root/tt-metal/models/experimental/mole/demo_checkpoints/etth1_DLinear_linear_sl336_pl96_td1_lr0.01_hd0.0_sd2021_MoLE_DLinear_ETTh1_ftM_sl336_ll336_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_DemoMatrix_0_32_1_f_mask_0.5_0.01_sd2021_hd0.0 \
  --checkpoint-file checkpoint.pth \
  --batch-size 8 \
  --warmup-iterations 100 \
  --measure-iterations 1000


**RLinear MoLE:**

python3 -m models.experimental.mole.demo.benchmark \
  --base-model-type rlinear \
  --dataset-dir /root/tt-metal/models/experimental/mole/demo_checkpoints \
  --dataset-file ETTh1.csv \
  --seq-len 336 --pred-len 96 --num-experts 4 \
  --checkpoint-dir /root/tt-metal/models/experimental/mole/demo_checkpoints/etth1_RLinear_mole_sl336_pl96_td4_lr0.005_hd0.2_sd2021_MoLE_RLinear_ETTh1_ftM_sl336_ll336_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_DemoMatrix_0_32_4_f_mask_0.5_0.005_sd2021_hd0.2 \
  --checkpoint-file checkpoint.pth \
  --batch-size 8 \
  --warmup-iterations 100 \
  --measure-iterations 1000


**RMLP MoLE:**

python3 -m models.experimental.mole.demo.benchmark \
  --base-model-type rmlp \
  --dataset-dir /root/tt-metal/models/experimental/mole/demo_checkpoints \
  --dataset-file ETTh1.csv \
  --seq-len 336 --pred-len 96 --num-experts 4 \
  --checkpoint-dir /root/tt-metal/models/experimental/mole/demo_checkpoints/etth1_RMLP_mole_sl336_pl96_td4_lr0.005_hd0.2_sd2021_MoLE_RMLP_ETTh1_ftM_sl336_ll336_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_DemoMatrix_0_32_4_f_mask_0.5_0.005_sd2021_hd0.2 \
  --checkpoint-file checkpoint.pth \
  --batch-size 8 \
  --warmup-iterations 100 \
  --measure-iterations 1000



### 3. Visualize router specialization

Generates a PNG showing how each expert is activated across time steps.

python3 -m models.experimental.mole.demo.specialization \
  --base-model-type dlinear \
  --dataset-dir /root/tt-metal/models/experimental/mole/demo_checkpoints \
  --dataset-file ETTh1.csv \
  --seq-len 336 --pred-len 96 --num-experts 4 \
  --checkpoint-dir /root/tt-metal/models/experimental/mole/demo_checkpoints/etth1_DLinear_mole_sl336_pl96_td4_lr0.01_hd0.2_sd2021_MoLE_DLinear_ETTh1_ftM_sl336_ll336_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_DemoMatrix_0_32_4_f_mask_0.5_0.01_sd2021_hd0.2 \
  --checkpoint-file checkpoint.pth \
  --image-path /tmp/mole_router_weights.png \
  --eval-batch-size 32
