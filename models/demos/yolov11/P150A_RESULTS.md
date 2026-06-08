# YOLOv11n on P150a (Blackhole) — 高速化結果まとめ

対象: `models/demos/yolov11`(ベースライン = ブランチ `yolo_bh_glx_tt_deploy`、元はWormhole向け)
ハード: P150a (Blackhole)、`/dev/tenstorrent/7`、計算グリッド 11×10 = **110コア**、DRAM 32GB/8バンク
日付: 2026-06-08 / 入力 640×640 検出、act/weight = bfloat8_b、Trace + 2CQ

---

## 1. 結果サマリ

| 構成 | FPS | iter | PCC | 備考 |
|---|---|---|---|---|
| **BS=1**（WHゲート除去のみ） | **368** | 2.72 ms | 0.9997 | 目標300達成。WH n150は234 FPS → 約1.57倍 |
| BS=1 device-only上限 | 364 | 2.75 ms | — | host入力prepは0.03ms(完全に隠蔽) |
| BS=2（初期) | 387 | 5.17 ms | 0.9949 | 単一デバイスバッチ実装 |
| BS=2 + reshard_all | 410 | 4.87 ms | 0.9949 | 各convを最適/フルグリッドへ |
| **BS=2 + reshard_all + bf16入力** | **535** | **3.74 ms** | 0.9949 | **デバイス上限に到達**。BS=1比 +45% |
| BS=4 / 8 | — | — | — | L1 OOM。全層DRAM退避が必要・低ROIで未実施 |

**結論: 目標300 FPSはBS=1で達成(368)。BS=2をデバイス上限535 FPSまで最適化した。**

---

## 2. 実行方法（レシピ）

### 2.1 環境の要点（ハマりどころ）
- リポジトリ内venv `python_env` のeditable `.pth`/finderは **`/tt-metal`** を指す。よって**リポジトリを `/tt-metal` にマウント**し、**`/tt-metal/python_env/bin/python`** を使う（imageの既定`python`だと `ttnn.__file__=None` でnamespace衝突）。
- **hugepageディレクトリのマウント必須**（無いと `Querying size for a host channel that does not exist` でデバイスオープン失敗）。
- `ultralytics==8.3.0` をvenvに導入済み（`common.py` が要求）。
- device 7 → NUMA node1（dev0-3=node0, dev4-7=node1）。`--cap-add SYS_NICE --cpuset-mems=1` でNUMA警告は消えるが性能差はほぼ無し。

### 2.2 ベースライン（BS=1, 目標達成確認）
```bash
cd /home/yito/work/tt-metal
docker run --rm --device /dev/tenstorrent/7 --cap-add SYS_NICE --cpuset-mems=1 \
  -v /dev/hugepages-1G:/dev/hugepages-1G -v $PWD:/tt-metal -w /tt-metal \
  -e TT_METAL_HOME=/tt-metal -e ARCH_NAME=blackhole \
  tt-metalium-dev:seamless_m4t_v2 \
  bash -lc '/tt-metal/python_env/bin/python -m pytest -s --disable-warnings \
    models/demos/yolov11/tests/perf/test_e2e_performant.py::test_e2e_performant'
# => FPS: 368, PCC 0.9997
```

### 2.3 BS=2 最適化（535 FPS / デバイス上限）
```bash
cd /home/yito/work/tt-metal
docker run --rm --device /dev/tenstorrent/7 --cap-add SYS_NICE --cpuset-mems=1 \
  -v /dev/hugepages-1G:/dev/hugepages-1G -v $PWD:/tt-metal -w /tt-metal \
  -e TT_METAL_HOME=/tt-metal -e ARCH_NAME=blackhole -e PYTHONPATH=/tt-metal \
  -e YOLO_RESHARD_ALL=1 -e YOLO_MAX_CORES=100 \
  -e YOLO_GRID_ROWS=10 -e YOLO_GRID_COLS=10 -e YOLO_GRID=10,8 \
  tt-metalium-dev:seamless_m4t_v2 \
  bash -lc '/tt-metal/python_env/bin/python -u models/demos/yolov11/bh_bs_sweep.py 2'
# [RESULT]=e2e(fp32入力 410), [E2E-BF16]=bf16入力 535, [DEVONLY]=上限 536
```
`bh_bs_sweep.py <BS,...>` は各バッチで PCC / e2e / bf16入力e2e / device-only / pipelined を計測する。

### 2.4 注意（デバイスwedge）
トレースキャプチャ失敗(OOM/clash)時に `close_device` がハングし `/dev/tenstorrent/7` を掴んだまま残る → 次回 `tt_tlb_alloc -12`。復旧:
```bash
docker kill <container>; kill -9 $(cat /proc/driver/tenstorrent/7/pids | head -1)
# /proc/driver/tenstorrent/7/pids が空になるまで確認
```

---

## 3. 高速化の3レバー（と効かなかったもの）

1. **conv自動再シャード `YOLO_RESHARD_ALL=1`**: 387→410（BS=2は活性化が大きく110コアを使い切れる）。**BS=1では逆効果**(368→361)なのでBS≥2限定。
2. **bf16入力（決定打）**: e2eの律速はデバイスでなく **host側 `from_torch` の fp32→bf16変換(1.14ms/iter, e2eの24%)** だった。入力を最初からbf16にするだけで消え、**410→535（デバイス上限）**。モデルは内部でbf16化するためPCC等価。
3. **コアグリッド 64→100/110**: BS=1では+1〜2%のみ(オーバーヘッド律速)。BS=2のバッチ＋reshardと組み合わせて初めて効く。

効かなかった/不要だったもの:
- **Pythonスレッドで入力prepをパイプライン化**: GIL競合で **292 FPSに悪化**。bf16入力が正解。
- **NUMA固定(SYS_NICE)**: BS=1で差ほぼ無し（host転送は律速でない）。

---

## 4. 加えた主なコード変更（すべて作業ツリー、env既定でBS=1は従来同等）

| ファイル | 変更 |
|---|---|
| `tests/perf/test_e2e_performant.py` | `@run_for_wormhole_b0()` をコメントアウト(BHで実行可) |
| `runner/performant_runner_infra.py` | `exit("Unsupported device")` 撤去; **単一デバイスのバッチ入力経路**(num_devices==1 で full `from_torch`); 入力グリッドを `YOLO_GRID` で可変 |
| `tt/common.py` | conv出力スライス `hw = batch*H*W`(バッチ脱落バグ修正); `sharded_concat`/`sharded_concat_2` をバッチ/グリッド対応(`_concat_shard_grid`); `is_detect` conv幾何のバッチ対応; env `YOLO_RESHARD_ALL` / `YOLO_NO_DBLBUF` / `YOLO_MAX_CORES` / `YOLO_GRID_ROWS,COLS` |
| `tt/ttnn_yolov11.py` | c2psa/detectへ `batch_size` 伝播; upsample前reshapeをバッチ対応(`sqrt(shape[2]//n)`); 大バッチ時(n>2)に入力ステムをDRAM経由 |
| `tt/ttnn_yolov11_c2psa.py` / `_psa.py` / `_attention.py` | `batch_size` スレッド; attention reshape `//batch_size`; 出力再構成を `permute→reshape` 順に修正(バッチをchannelに畳まない) |
| `tt/ttnn_yolov11_detect.py` | detectヘッド全面バッチ対応(全convへbatch伝播; squeeze廃止しdim0にバッチ保持; DFL出力 `(bs,4,...)` でアンカーをper-imageにbroadcast) |
| `bh_bs_sweep.py` / `bs2_debug.py` | 計測・デバッグ補助(新規) |

---

## 5. さらに上を目指すなら
- **複数P150aのData Parallel**: ベースラインの `test_e2e_performant_dp` を今回同様BH対応させれば台数ほぼ線形(各カード 368@BS1 / 535@BS2 相当)。単一カードの上限は535 FPS。
- **BS≥4**: 全層のスキップテンソル＋conv活性化をDRAM退避すれば動くが、per-imageが頭打ちのため利得は限定的(低ROI)。
