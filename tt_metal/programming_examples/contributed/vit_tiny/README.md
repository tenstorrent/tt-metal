# ViT Tiny on TT-Metalium

ViT Tiny (`vit_tiny_patch16_224.augreg_in21k_ft_in1k`) の推論を TT-Metalium の低レベル C++ API のみで実装したプログラミングサンプルです。TT-NN は使用しません。

## Model Architecture

| Parameter | Value |
|---|---|
| Model | ViT-Tiny/16 (timm) |
| Image size | 224 x 224 |
| Patch size | 16 x 16 |
| Num patches | 196 (+1 CLS = 197) |
| Embed dim | 192 |
| Depth | 12 |
| Num heads | 3 |
| Head dim | 64 |
| MLP dim | 768 |
| Num classes | 1000 |
| Activation | GELU |
| Norm | LayerNorm (eps=1e-6) |

### Inference Flow

```
Image [3, 224, 224]
  |
  v
Patch Embedding
  Unfold to patches [224, 768]    (host)
  Linear projection [224, 192]    (matmul, 7 cores)
  + bias                          (eltwise_add, 7 cores)
  + CLS token                     (eltwise_add, 7 cores)
  + position embedding            (eltwise_add, 7 cores)
  |
  v
Transformer Block x12
  LayerNorm                       (layernorm, 7 cores)
  Multi-Head Self-Attention
    QKV projection [224, 576]     (matmul, 7 cores)
    + QKV bias                    (eltwise_add, 7 cores)
    Column slice Q, K, V          (column_slice)
    Per-head (x3):
      Column slice Q_h, K_h, V_h  (column_slice)
      Transpose K_h               (transpose_2d)
      Q_h @ K_h^T                 (matmul, 7 cores)
      Scale (x 0.125)             (eltwise_mul, 7 cores)
      Softmax                     (softmax, 7 cores)
      Attn @ V_h                  (matmul, 7 cores)
      Write to concat buf         (column_write)
    Output projection [224, 192]  (matmul, 7 cores)
    + proj bias                   (eltwise_add, 7 cores)
  Residual add                    (eltwise_add, 7 cores)
  LayerNorm                       (layernorm, 7 cores)
  MLP
    FC1 [224, 768]                (matmul, 7 cores)
    + bias + GELU                 (eltwise_add + gelu, 7 cores)
    FC2 [224, 192]                (matmul, 7 cores)
    + bias                        (eltwise_add, 7 cores)
  Residual add                    (eltwise_add, 7 cores)
  |
  v
Final LayerNorm                   (layernorm, 7 cores)
  |
  v
CLS token extraction [1, 192]    (host, 唯一のデバイス→ホスト往復)
  |
  v
Classification head [1, 1000]     (matmul + eltwise_add)
  |
  v
Top-5 predictions
```

### Tile Dimensions

全データは 32x32 タイル形式で処理されます。主要な次元のタイル数:

| Dimension | Padded | Tiles |
|---|---|---|
| seq_len (197 -> 224) | 224 | 7 |
| embed_dim (192) | 192 | 6 |
| mlp_dim (768) | 768 | 24 |
| head_dim (64) | 64 | 2 |
| qkv_dim (576) | 576 | 18 |
| num_classes (1000 -> 1024) | 1024 | 32 |

## Source Code Structure

```
vit_tiny/
├── vit_tiny.cpp                    # メインプログラム (op テスト + E2E 推論)
├── CMakeLists.txt                  # ビルド設定
├── README.md
│
├── ops/                            # ホスト側 op ラッパー (カーネル起動)
│   ├── common.hpp                  #   共通ユーティリティ, MeshContext, バッファ管理
│   ├── matmul.hpp                  #   行列乗算 (マルチコア, 行分散)
│   ├── eltwise_add.hpp             #   要素和 (マルチコア, タイル分散)
│   ├── eltwise_mul.hpp             #   要素積 (マルチコア, タイル分散)
│   ├── gelu.hpp                    #   GELU 活性化 (マルチコア, タイル分散)
│   ├── softmax.hpp                 #   Softmax (マルチコア, 行分散)
│   ├── layernorm.hpp               #   LayerNorm (マルチコア, 行分散)
│   ├── column_slice.hpp            #   列スライス抽出 (単一コア)
│   ├── column_write.hpp            #   列オフセット書き込み (単一コア)
│   └── transpose.hpp               #   2D 行列転置 (単一コア)
│
├── model/                          # モデル構造 (op の組み合わせ)
│   ├── vit_model.hpp               #   ViT 全体: patch_embed → blocks → LN → head
│   ├── patch_embed.hpp             #   パッチ埋め込み: unfold + 射影 + CLS + pos_embed
│   ├── transformer_block.hpp       #   Transformer ブロック: LN → Attn → Residual → LN → MLP → Residual
│   ├── attention.hpp               #   Multi-Head Self-Attention (全デバイス演算)
│   └── mlp.hpp                     #   MLP: FC1 → GELU → FC2
│
├── weights/                        # 重み管理
│   ├── export_weights.py           #   timm → binary エクスポートスクリプト
│   └── weight_loader.hpp           #   binary → tilize → DRAM アップロード
│
└── kernels/                        # デバイスカーネル (RISC-V)
    ├── compute/                    #   計算カーネル (Unpack → Math/SFPU → Pack)
    │   ├── matmul_compute.cpp      #     matmul_tiles
    │   ├── eltwise_add_compute.cpp #     add_tiles
    │   ├── eltwise_mul_compute.cpp #     mul_tiles
    │   ├── gelu_compute.cpp        #     gelu
    │   ├── softmax_compute.cpp     #     max, sub, exp, sum, recip, mul
    │   ├── layernorm_compute.cpp   #     mean, variance, normalize, scale+shift
    │   └── transpose_compute.cpp   #     transpose_wh_tile
    │
    └── dataflow/                   #   データ移動カーネル (NoC DMA)
        ├── reader_matmul_multicore.cpp     # A の部分行 + B 全体を読み取り
        ├── writer_matmul_multicore.cpp     # タイルオフセット付き書き込み
        ├── reader_binary_multicore.cpp     # 2 入力タイルオフセット読み取り
        ├── reader_unary_multicore.cpp      # 1 入力タイルオフセット読み取り
        ├── writer_unary_multicore.cpp      # タイルオフセット付き書き込み
        ├── reader_softmax_multicore.cpp    # 行オフセット + scaler CB 準備
        ├── reader_layernorm_multicore.cpp  # 行オフセット + eps/gamma/beta CB 準備
        ├── reader_column_slice.cpp         # 列オフセット付き読み取り
        ├── reader_transpose.cpp            # 転置順序でタイル読み取り
        ├── reader_unary_to_out.cpp         # cb_out 直接書き込み (compute-free op用)
        ├── writer_column_write.cpp         # 列オフセット付き書き込み
        ├── reader_matmul.cpp               # (単一コア版, 未使用)
        ├── writer_matmul.cpp               # (単一コア版, 未使用)
        ├── reader_binary.cpp               # (単一コア版, 未使用)
        ├── reader_unary.cpp                # (単一コア版, 未使用)
        ├── writer_unary.cpp                # 列スライス/転置用ライター
        ├── reader_softmax.cpp              # (単一コア版, 未使用)
        └── reader_layernorm.cpp            # (単一コア版, 未使用)
```

### Multicore Strategy

全 op は最大 7 コアで並列実行されます (`CoreRange({0,0}, {N-1, 0})`)。op ごとに逐次実行するため、同時使用は最大 7 コアです。

| Op | Parallelism | Distribution |
|---|---|---|
| matmul | 7 cores | 出力行分散 (各コア Mt/7 行) |
| eltwise_add / mul | 7 cores | タイル分散 (各コア n_tiles/7 タイル) |
| gelu | 7 cores | タイル分散 |
| softmax | 7 cores | 行分散 |
| layernorm | 7 cores | 行分散 |
| column_slice | 1 core | データ移動のみ |
| column_write | 1 core | データ移動のみ |
| transpose_2d | 1 core | 小行列 (2x7 タイル) |

### Host/Device Boundary

推論中のホスト ↔ デバイス間データ転送:

| Location | Direction | Purpose |
|---|---|---|
| Patch embedding | Host → Device | Unfold された画像パッチの書き込み (1 回) |
| CLS extraction | Device → Host | 最終 LayerNorm 後の CLS トークン読み出し (1 回) |
| Classification head | Host → Device / Device → Host | CLS 特徴の再アップロード + logits 読み出し |

Attention 内の QKV 分割、K 転置、スコアスケーリング、ヘッド結合は全てデバイス上で完結します。

## Build

```bash
cd tt-metal

# vit_tiny のみビルド
ninja -C build_Release metal_example_vit_tiny

# 全プログラミングサンプルをビルド
./build_metal.sh --build-programming-examples
```

## Weight Export

timm から重みをエクスポートします。Python 環境に `timm`, `torch`, `torchvision` が必要です。

```bash
cd tt-metal

# 重みエクスポート
python tt_metal/programming_examples/vit_tiny/weights/export_weights.py \
    --output-dir vit_tiny_weights

# テスト画像もエクスポート (任意の JPEG/PNG を指定可能)
python tt_metal/programming_examples/vit_tiny/weights/export_weights.py \
    --output-dir vit_tiny_weights \
    --export-image \
    --test-image path/to/image.jpg
```

出力ファイル:
- `vit_tiny_weights/*.bin` — float32 バイナリ形式の重み (154 ファイル)
- `vit_tiny_weights/test_image.bin` — 前処理済み画像 [3, 224, 224] float32
- `vit_tiny_weights/reference_logits.bin` — PyTorch の参照出力 (PCC 比較用)

## Run

### Op Unit Tests

引数なしで実行すると、全 op (matmul, eltwise_add, gelu, softmax, layernorm, eltwise_mul, column_slice, transpose, column_write) の単体テストを実行します。

```bash
export TT_METAL_HOME=$(pwd)
./build_Release/programming_examples/metal_example_vit_tiny
```

### E2E Inference

重みディレクトリとテスト画像を指定して推論を実行します。

```bash
export TT_METAL_HOME=$(pwd)
./build_Release/programming_examples/metal_example_vit_tiny \
    vit_tiny_weights \
    vit_tiny_weights/test_image.bin
```

### Docker

```bash
sudo docker run --rm \
    -v /home:/home \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    --device /dev/tenstorrent/2 \
    -w /path/to/tt-metal \
    <container_image> \
    bash -c 'export TT_METAL_HOME=$(pwd) && \
             ./build_Release/programming_examples/metal_example_vit_tiny \
             vit_tiny_weights vit_tiny_weights/test_image.bin'
```

### Expected Output

```
Top-5 predictions:
  Class 599: 7.9688
  Class 506: 7.4375
  Class 646: 7.3750
  Class 815: 6.7500
  Class 310: 6.5000

Logits PCC vs PyTorch reference: 0.9501485
```

Op テスト PCC:

| Op | PCC |
|---|---|
| matmul | 0.9999 |
| eltwise_add | 0.999999 |
| gelu | 0.9998 |
| softmax | 0.9993 |
| layernorm | 0.99998 |
| eltwise_mul | 0.9999 |
| column_slice | 1.0 |
| transpose | 1.0 |
| column_write | 1.0 |
