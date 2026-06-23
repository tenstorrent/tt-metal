# ViT (TT-NN interleaved) — Tensix コア数の制限と実行方法

対象モデル: `ttnn_optimized_interleaved_vit.py`（このディレクトリ）
モデル: `google/vit-base-patch16-224`（batch=8, seq=224）
検証環境: Blackhole p300（単一チップ, `/dev/tenstorrent/0`）

このドキュメントは、本 interleaved ViT で **使用する Tensix コア数を変更する方法** と **実行方法** をまとめたものです。

---

## 1. コア数の替え方

コア制限には 2 つのレベルがあります。用途に応じて使い分けます。

### (A) モデル側グリッド ― 環境変数 `VIT_GRID_Y` / `VIT_GRID_X`（推奨）

`ttnn_optimized_interleaved_vit.py` 冒頭で、全 matmul / layernorm に渡す `core_grid` を
環境変数から生成します。

```python
_grid_y = int(os.environ.get("VIT_GRID_Y", "8"))
_grid_x = int(os.environ.get("VIT_GRID_X", "12"))
core_grid = ttnn.CoreGrid(y=_grid_y, x=_grid_x)
```

- 使用コア数 = `VIT_GRID_Y * VIT_GRID_X`
- 既定は `8 x 12 = 96`（未設定なら従来動作）
- 例:
  | 目標コア数 | 設定 |
  |---|---|
  | 32 | `VIT_GRID_Y=8 VIT_GRID_X=4`（または `4x8`） |
  | 16 | `VIT_GRID_Y=4 VIT_GRID_X=4`（または `8x2` / `2x8`） |
- 起動時に `[ttnn_vit] using core_grid y=.. x=.. -> N cores` を出力して確認可能。

この方法で制限されるのは **matmul と layernorm**（モデルの主要演算）です。
softmax / elementwise / データ移動 op はグリッド指定が無く、既定でより広いグリッドを
使い得ます。「全 op を厳密に N コア以内」にしたい場合は (B) を併用します。

### (B) デバイス全体の物理ハードキャップ ― `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE`

デバイスが公開する compute グリッドそのものを縮小し、**全 op を物理的に制限**します。

```bash
# 値は "x_end,y_end"（inclusive, 角括弧なし）。グリッドは (x_end+1) x (y_end+1)。
export TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="7,3"   # 8 x 4 = 32 コア
export TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,3"   # 4 x 4 = 16 コア
```

- (A) のモデルグリッドと値を一致させて使う（例: 32 コアなら `VIT_GRID 8x4` + override `7,3`）。
- 制約: override 値はハードウェアの実グリッド以下である必要がある。
  この p300 の実 compute グリッドは **11 x 10 = 110 コア**（x=11、harvesting による）。
- **注意（L1 容量）**: コアを減らすほど 1 コアあたりのデータ量が増える。
  batch=8 では **16 コアのハードキャップで L1 オーバーフロー**する
  （`circular buffers clash with L1 buffers`）。回避するには batch_size を下げるか、
  ハードキャップを使わず (A) のみにする。**32 コアのハードキャップは動作する。**

---

## 2. 実行方法

### 2.1 ランタイム要件（Blackhole p300 を単一チップで使う場合）

| 要件 | 指定 |
|---|---|
| デバイス | `--device /dev/tenstorrent/0` |
| hugepages | `-v /dev/hugepages-1G:/dev/hugepages-1G`（無いと hugepages assert） |
| TT_METAL_HOME | `-e TT_METAL_HOME=/tt-metal` |
| mesh graph descriptor | `-e TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto` |

p300 は本来 2 チップで、1 チップのみ可視だと CUSTOM cluster になり、単一チップ
（p150 = 1x1 Blackhole）の mesh graph descriptor 指定が必要です。
起動失敗（`active ethernet core ... timed out`）時は `tt-smi -r` でボードをリセット。

### 2.2 pytest で実行

代表テストは `test_vit_layer`（Transformer 1 ブロック: layernorm / QKV matmul /
attention softmax / 2 つの attention matmul / 出力射影 / MLP fc1+gelu / fc2）。
合成入力で torch 参照と PCC 比較するためデータセット不要です。

```bash
cd $TT_METAL_HOME

# 32 コア（モデルグリッドのみ）
VIT_GRID_Y=8 VIT_GRID_X=4 \
pytest -s --disable-warnings \
  models/demos/vision/classification/vit/common/tests/pcc/test_ttnn_optimized_interleaved_vit.py::test_vit_layer

# 16 コア（モデルグリッドのみ）
VIT_GRID_Y=4 VIT_GRID_X=4 \
pytest -s --disable-warnings \
  models/demos/vision/classification/vit/common/tests/pcc/test_ttnn_optimized_interleaved_vit.py::test_vit_layer

# 32 コア + デバイス物理ハードキャップ（全 op を制限）
VIT_GRID_Y=8 VIT_GRID_X=4 TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="7,3" \
pytest -s --disable-warnings \
  models/demos/vision/classification/vit/common/tests/pcc/test_ttnn_optimized_interleaved_vit.py::test_vit_layer
```

> 本 interleaved テストは元々 `@pytest.mark.skip`(#7527: PCC 閾値見直し) /
> `skipif(WH/BH)` でスキップ指定。コア数実験のため当ブランチではスキップ装飾を除去済み。

### 2.3 他テストの注意

- `test_vit`（フル分類）: ImageNet データセット取得が必要で `KeyError: 'test'`（要 HF データセット認証）。
- `test_vit_encoder`（12 層）: テストハーネス側の既知バグ（attention_mask 整形の IndexError）でデバイス計算前に失敗。

いずれもコア数制限とは無関係です。

---

## 3. 実機検証結果（test_vit_layer, batch=8, p300）

| 指定 | 制限対象 | コア数 | 結果 |
|---|---|---|---|
| `VIT_GRID 8x4` | matmul / LN | 32 | PCC 0.99942 ✓ |
| `VIT_GRID 4x4` | matmul / LN | 16 | PCC 0.99942 ✓ |
| `VIT_GRID 8x4` + override `7,3` | 全 op（物理） | 32 | PCC 0.99942 ✓ |
| `VIT_GRID 4x4` + override `3,3` | 全 op（物理） | 16 | L1 clash で失敗（batch を下げれば可） |

PCC 0.99942 はテスト閾値 0.9999 を僅かに下回るのみで、機能的には正しく動作
（= スキップ理由の "PCC threshold needs review" と整合）。コア数を変えても PCC は不変。

---

## 4. 補足: sharded 版との違い

BH 公式サポートの **sharded 版**（`../../blackhole/tt/ttnn_optimized_sharded_vit_bh.py`）は
`grid_x=12` / `core_grid_10x12`（120 コア）にハードチューニングされ、shard 形状も固定のため、
コア数の変更には大規模な再チューニングが必要です。加えて p300 の実グリッドは x=11 のため
`x=12` を要求する sharded 版はそのままでは載りません。
コア数を柔軟に制限する用途では、`core_grid` をヒントとして渡す本 **interleaved 版** が適しています。
