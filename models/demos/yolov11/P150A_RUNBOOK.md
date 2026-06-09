# 手順書: ブランチ受領 → tt-metalコンテナビルド → YOLOv11n 性能テスト (P150a / Blackhole)

ブランチ `yito/yolo11n_p150` を受け取り、ゼロから tt-metal をビルドして YOLOv11n の
パフォーマンステストを実行するまでの一連の手順。

> 前提ハード: Tenstorrent **Blackhole P150a** が1枚以上挿さったLinuxホスト。
> 結果の詳細は同ディレクトリの `P150A_RESULTS.md` を参照。

---

## 0. パラメータ（環境に合わせて設定）

```bash
export REPO=$HOME/work/tt-metal          # リポジトリのクローン先（ホスト側）
export IMG=tt-metalium-dev:yolo11n       # ビルドするdevイメージのタグ
export DEV_ID=0                          # 使用するデバイス（/dev/tenstorrent/<ID>）
export NUMA_NODE=0                       # DEV_ID が属する NUMA ノード（後述で確認）
```

---

## 1. ホスト前提条件の確認

```bash
# 1-1. デバイスが見えるか
ls /dev/tenstorrent/            # 0,1,... が見えること

# 1-2. Docker が使えるか（dockerグループ所属）
docker ps

# 1-3. 1GB hugepage がデバイス分用意されているか（★必須）
ls /dev/hugepages-1G/           # device_<ID>_tenstorrent / tenstorrent が並ぶこと
#   無い場合は Tenstorrent KMD/セットアップ(tt-installer 等)で hugepage を構成する。
#   これが無いと後段でデバイスオープンが
#   "Querying size for a host channel that does not exist" で失敗する。

# 1-4. DEV_ID の NUMA ノードを確認（任意・性能用）
#   PCIベンダ 0x1e52 が Tenstorrent。デバイス順に node を確認:
for d in /sys/bus/pci/devices/*; do
  [ "$(cat $d/vendor 2>/dev/null)" = "0x1e52" ] && echo "$d numa=$(cat $d/numa_node)"
done
#   例: 8枚構成では dev0-3=node0, dev4-7=node1。DEV_ID に対応する node を NUMA_NODE に設定。
```

---

## 2. ブランチを受け取る

```bash
git clone git@github.com:tenstorrent/tt-metal.git "$REPO"
cd "$REPO"
git checkout yito/yolo11n_p150
git submodule update --init --recursive      # tt-llk 等のサブモジュール
```

---

## 3. dev イメージをビルド

`dev` ターゲットは「ツールチェーン + uv/Python」だけのイメージ（tt-metal本体は含まない）。

```bash
cd "$REPO"
docker build -f dockerfile/Dockerfile --target dev \
  --build-arg UBUNTU_VERSION=22.04 --build-arg PYTHON_VERSION=3.10 \
  -t "$IMG" .
```

---

## 4. コンテナ内で tt-metal をビルド + venv 作成 + 依存導入

**重要:** リポジトリ内 `python_env` の editable パスは **`/tt-metal`** を前提にするため、
ビルド・実行とも **リポジトリを `/tt-metal` にマウント** し、コンテナ内venv
**`/tt-metal/python_env/bin/python`** を使う（イメージ既定の `python` は使わない）。

```bash
docker run --rm \
  -v "$REPO":/tt-metal -w /tt-metal \
  -e TT_METAL_HOME=/tt-metal -e ARCH_NAME=blackhole \
  "$IMG" bash -lc '
    set -e
    ./build_metal.sh --enable-ccache            # C++ 本体ビルド（初回 ~30-60分）。build/ が生成される
    ./create_venv.sh                            # python_env を作成（Python3.10, editable install）
    uv pip install --python /tt-metal/python_env/bin/python "ultralytics==8.3.0"  # common.py が要求
  '
```
> 生成物 `build/` と `python_env/` はマウント経由でホスト側 `$REPO` に残る（次回以降は再ビルド不要）。

### 4-1. 動作確認（任意）
```bash
docker run --rm --device /dev/tenstorrent/$DEV_ID \
  -v /dev/hugepages-1G:/dev/hugepages-1G -v "$REPO":/tt-metal -w /tt-metal \
  -e TT_METAL_HOME=/tt-metal -e ARCH_NAME=blackhole \
  "$IMG" bash -lc '/tt-metal/python_env/bin/python -c "import ttnn; from ttnn.device import Arch; print(ttnn.__file__)"'
# => /tt-metal/ttnn/ttnn/__init__.py と表示されればOK（None ならマウント/パス誤り）
```

---

## 5. YOLOv11n 性能テストを実行

共通の docker オプション（デバイス + hugepage + 同一パスマウント + venv）を使う。

### 5-1. ベースライン (BS=1, 目標300 FPS 達成確認)
```bash
docker run --rm --device /dev/tenstorrent/$DEV_ID --cap-add SYS_NICE --cpuset-mems=$NUMA_NODE \
  -v /dev/hugepages-1G:/dev/hugepages-1G -v "$REPO":/tt-metal -w /tt-metal \
  -e TT_METAL_HOME=/tt-metal -e ARCH_NAME=blackhole \
  "$IMG" bash -lc '/tt-metal/python_env/bin/python -m pytest -s --disable-warnings \
    models/demos/yolov11/tests/perf/test_e2e_performant.py::test_e2e_performant'
# 期待: "FPS: 368" 付近, PCC 0.9997, PASSED （初回はカーネルコンパイルで数分）
```

### 5-2. BS=2 最適化（reshard + grid + bf16入力 → デバイス上限 ~535 FPS）
```bash
docker run --rm --device /dev/tenstorrent/$DEV_ID --cap-add SYS_NICE --cpuset-mems=$NUMA_NODE \
  -v /dev/hugepages-1G:/dev/hugepages-1G -v "$REPO":/tt-metal -w /tt-metal \
  -e TT_METAL_HOME=/tt-metal -e ARCH_NAME=blackhole -e PYTHONPATH=/tt-metal \
  -e YOLO_RESHARD_ALL=1 -e YOLO_MAX_CORES=100 \
  -e YOLO_GRID_ROWS=10 -e YOLO_GRID_COLS=10 -e YOLO_GRID=10,8 \
  "$IMG" bash -lc '/tt-metal/python_env/bin/python -u models/demos/yolov11/bh_bs_sweep.py 2'
# 出力の意味:
#   [RESULT]    = e2e (fp32入力)       ~410 FPS
#   [E2E-BF16]  = e2e (bf16入力)       ~535 FPS  ← 実運用で狙う値
#   [DEVONLY]   = デバイス上限         ~536 FPS
#   [PIPELINED] = Pythonスレッド版     ~290 FPS（GIL競合で悪化。参考）
```

### 5-3.（任意）デバイスカーネル上限の測定
```bash
docker run --rm --device /dev/tenstorrent/$DEV_ID \
  -v /dev/hugepages-1G:/dev/hugepages-1G -v "$REPO":/tt-metal -w /tt-metal \
  -e TT_METAL_HOME=/tt-metal -e ARCH_NAME=blackhole \
  "$IMG" bash -lc '/tt-metal/python_env/bin/python -m pytest -s --disable-warnings \
    models/demos/yolov11/tests/perf/test_perf.py::test_perf_device_bare_metal_yolov11'
```

---

## 6. トラブルシュート

| 症状 | 原因 / 対処 |
|---|---|
| `ttnn.__file__` が `None` / `No module named 'ttnn.device'` | リポジトリを `/tt-metal` にマウントし、`/tt-metal/python_env/bin/python` を使う |
| `Querying size for a host channel that does not exist` | `-v /dev/hugepages-1G:/dev/hugepages-1G` を付け忘れ。ホストの hugepage 設定も確認 |
| `No module named 'ultralytics'` | 手順4のvenvへの `uv pip install ultralytics==8.3.0` を実行 |
| `tt_tlb_alloc ... -12` / 次回デバイスオープン失敗 | 前回のトレースキャプチャ失敗でプロセスがデバイスを掴んだまま残留。下記で解放: |
| | `docker kill <container>; kill -9 $(cat /proc/driver/tenstorrent/$DEV_ID/pids \| head -1)`<br>`/proc/driver/tenstorrent/$DEV_ID/pids` が空になるまで確認 |
| `pre-commit not found`（コミット時） | `git commit --no-verify`（フック未導入環境） |

---

## 7. 補足（高速化レバーの意味）

- `YOLO_RESHARD_ALL=1`: 各convを最適/フルグリッド(～110コア)へ再シャード。**BS≥2で有効**（BS=1では微減）。
- `YOLO_GRID=10,8`: 入力L1シャードを80コアに（バッチ時のシャード数がコア数を超えない最大の約数）。
- **bf16入力**: e2e律速だった host側 `from_torch` の fp32→bf16変換を消す決定打（`bh_bs_sweep.py` が自動計測）。
- BS≥4 は L1 を全層DRAM退避しないと動かず、利得も限定的（`P150A_RESULTS.md §5`）。
- 複数カードでの増強は Data Parallel（各カード上限相当、ほぼ線形）が本命。
