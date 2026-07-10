# Qwen3.6-27B — Batched Decode (batch>1) 設計メモ（方針B）

対象: `models/demos/blackhole/qwen36/`（main / worktree `yito/qwen36_27b_port_to_main`）
出典: prototype branch `yito/qwen36_27b_p300x2_tp` の batched-decode 最適化を main へ移植（Task #2）。
方針: **B（main 既存 python recurrent decode 経路で `max_batch_size>1` を有効化）**。C++ カーネル（#3）非依存。

---

## 0. 前提（調査で確定した事実）

- **decode trace は既に存在**（B=1）。`demo/text_demo.py:577-593`（`begin/end_trace_capture` + `execute_trace`）、contract 経路 `ttnn_decode_forward`(model.py:1801)+`prepare_inputs_decode`(1790)、vLLM warmup。→ **trace 機構は移植不要**。batched 化は「同じ機構を B>1 の forward で capture する」だけ。
- **decode forward は既に B 汎用に記述済み**:
  - GDN `gdn/tp.py:356 forward_decode` — 全 slice/reshape が `B` 基準。再帰は `(B,1,Nv,Dk)`。
  - Attention `attention/tp.py:210 forward_decode` — `B` 基準、per-user `cur_pos_tt`（位置テンソル）、`page_table`、SDPA-decode 対応済み。
- **再帰カーネルは B=8 でスケール可能**: `recurrent_gated_delta_rule_decode_ttnn`（`models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py`）は `[B*H,1,K]@[B*H,K,V]` で B·H を matmul M 次元へ畳み込み、グリッドを tile 数で自動選択（per_core_M は L1 安全域にクランプ）。branch の flattened-head カーネルの **B≤9(110コア)制約は無い**。B=8, Nv_tp=12 → B·H=96 行(3 tile) で問題なし。
- **in-place fixed-buffer state（#4）は概ね実装済み**: `_stable_state` / `reset_state_inplace` / `conv_carry` / `_zero_conv0`（gdn/tp.py:144-216, 419-425）。#4 は別途「差分確認のみ」で足りる見込み。

つまり **真のギャップは 2 点だけ**:
1. **B>1 の静的 batched decode** を serving 経路に「配線」する（shape は既に B 対応。B=1 ハードコードを外す）。
2. **continuous batching / per-slot state reseed**（1 スロットだけ GDN rec/conv + KV を reset して途中参加/離脱）。

---

## 1. ギャップ1: 静的 batched decode（B>1）の配線

### 1.1 変更箇所

| # | file:line | 現状 | 変更 |
|---|-----------|------|------|
| a | `tt/model_config.py:36` | `max_batch_size=1` (default) | default 維持。B は呼び出し側から渡す。`_init_tp_config` に `B*Nv_tp` の core 見積り assert を追加検討 |
| b | `tt/model.py:140` `from_pretrained(..., max_batch_size=1)` | B=1 | 呼び出し側から B を透過（そのまま OK、値だけ >1 を許可） |
| c | `tt/model.py:1475` `allocate_kv_caches(..., batch_size=1)` | B=1 ハードコード | 引数 B を実際に使う。KV/GDN 外部 state を `[B,...]` で確保 |
| d | `tt/model.py:1417` `reset_state(batch_size=1)` | B=1 | B 透過 |
| e | `tt/qwen36_vllm.py:129` `allocate_kv_caches(..., batch_size=1)` | B=1 | `max_batch_size` を透過（ギャップ2 と一体） |
| f | `tt/model.py:220 decode_tp(token_id, pos)` | scalar token + scalar pos（demo oracle） | `token_ids: list[int]`, `positions: list[int]` を受ける batched 版を追加（下記 1.2） |
| g | `tt/model.py:254 generate_tp` | 1 本の greedy | B 本同時 greedy（各系列 EOS 管理）。デモ検証用 |

> 注: `prefill_tp`(177) と TP prefill(1581) の `assert B==1` は **prefill 側**。batched decode は「1 本ずつ prefill → 全スロットまとめて decode」でも成立するので、prefill の B=1 制約は**このタスクでは触らない**（別タスク）。continuous batching では各スロットを個別 prefill してスロットへ書き込む。

### 1.2 `decode_tp` batched 版（デモ oracle 経路）

現状 `decode_tp(token_id, pos)` は `torch.tensor([[int(token_id)]])` → `x=[1,1,1,dim_frac]`。
batched 版:
- 入力 `tok = torch.tensor([[t] for t in token_ids], int32)` → `[B,1]` → embed → `reshape (1,1,B,dim_frac)`（decode の batch 次元は dim=2、既存コードと整合: attention/gdn は `[1,B,dim]` へ reshape）。
- `cur_pos_tt = ttnn.from_torch(torch.tensor(positions, int32))`（per-user 位置。attention SDPA-decode / paged_update_cache が既に位置テンソルを受ける）。
- `rot_mats_decode` を per-user 位置で生成（現状 `torch.tensor([pos])` → `torch.tensor(positions)`）。
- logits 読み出し: 現状 `lt[0].reshape(-1)[:vocab]`（1 本）→ `[B, vocab]` を返す。

### 1.3 検証ポイント（実機）

- SDPA-decode grid cap（attention/tp.py:239-243, 64 コア上限）が B>1 でオーバーフローしないか。B が増えると SDPA-decode は batch 方向に並ぶので `max_cores_per_head_batch` 相当の再検討が要る可能性。
- `apply_partial_rope_decode(..., B, ...)`（attention/tp.py:236-237）は既に B 引数を取る → B>1 で shape 検証。
- 再帰 `recurrent_gated_delta_rule_decode_ttnn` の per_core_M L1 クランプが B·H=96 で成立するか（設計上 OK、実測要）。
- fp32 rec_state（default）のメモリ: `[B,Nv_tp,Dk,Dv]=[8,12,128,128]` fp32 ≈ 50MB/chip。KV `[B,1,max_seq,HD]` × NKV。B=8 で DRAM 収まるか。

---

## 2. ギャップ2: continuous batching / per-slot state reseed

branch: 「staggered-join reseed」「continuous-batching conv_hist reseed」= 走行中の他スロットを保持したまま 1 スロットだけ新規系列で reseed。

### 2.1 現状（全て whole-batch）
- `gdn/tp.py:152 reset_state` / `:183 reset_state_inplace` — conv/rec 全体を zero（batch-index 引数なし）。
- `attention/tp.py:120 reset_state` — KV 全体 zero。
- `model.py:1386 _reset_gdn_state_for_new_sequence` / `:1419 _reset_dn_state_inplace` — 全体。
- vLLM: `max_concurrency=1` 明言（qwen36_vllm.py docstring）。

### 2.2 追加が必要な API
- `TPGatedDeltaNet.reset_slot_inplace(b: int)`: `rec_state[b]` と各 `conv_states[k][:, b, :]` だけ zero（`ttnn.copy` を slot slice へ）。
- `TPAttention.reset_slot_inplace(b: int)`: `k/v_caches[h][b]`（or paged: 該当 page_table 行）だけ zero。
- `Qwen36Model.reset_slot(b)`: 上記を全 layer へ。
- prefill→スロット書き込み: 単一系列 prefill 結果（GDN rec/conv, KV）を batched buffer の slot b へ `ttnn.copy`。
- generator/vLLM: `max_num_seqs>1`、per-request `cur_pos`/page_table 行、スロット割当・解放。

### 2.3 trace との整合（重要）
- `_stable_state=True` で rec_state/KV のアドレスを固定 → decode trace は B 固定で 1 度だけ capture。
- per-slot reseed は **trace 外**の `ttnn.copy`（固定バッファへの部分書き込み）で行えばアドレス不変 → 既存 trace を維持したままスロット差し替え可能。branch の staggered-join と同型。

---

## 3. 段階的インクリメント（PR 粒度の提案）

1. **Step 1（最小・検証容易）**: `decode_tp` batched 版 + `generate_tp` B 本 greedy + `allocate/reset` の B 透過。デモ oracle 経路のみ。B=2 で 2 プロンプト同時生成が単発 B=1×2 と一致することを host-PCC/greedy 一致で検証（branch の `tp_batch2.py` 相当）。
2. **Step 2**: batched decode trace（Step 1 の forward を `begin/execute_trace` で B 固定 capture）。B=8 スループット計測。
3. **Step 3**: per-slot reseed API（2.2）+ 単発テスト（走行中 slot0 保持で slot1 reseed が正しいこと）。
4. **Step 4**: vLLM `max_num_seqs>1` 配線（qwen36_vllm.py / generator_interface.py）。continuous batching E2E。

各 Step は独立 PR 可能。Step 1〜2 が「batch=8 + trace」の中核、Step 3〜4 が continuous batching。

---

## 4. テスト
- 新規 `tests/test_batched_decode_tp.py`: B=2 で「2 系列同時 decode」==「各系列 B=1 decode」を greedy トークン一致＋ logits PCC で検証。既存 `tests/test_generate_tp.py` の隣に。
- `tests/pcc_thresholds.json` に batched decode の閾値追加。
- 実機: device 2（空き、単一 Blackhole=P150 相当。TP=4 が要るなら P300x2 全4chip、他デバイス使用状況を要確認）。TP=4 実機は `MESH_DEVICE=P150x4` 相当の 4chip 確保が前提。

## 5. 未解決 / 要判断
- B の上限: 再帰は B=8 まで設計上 OK だが、SDPA-decode grid・DRAM 実測が上限を決める。branch は B≤9。まず B=8 目標。
- Step 4（vLLM continuous batching）は工数大。#3（C++ decode カーネル）後に回す判断もあり得る。
- prefill の B=1 制約（177/1581）は本タスク対象外。continuous batching では per-slot 個別 prefill で回避。
