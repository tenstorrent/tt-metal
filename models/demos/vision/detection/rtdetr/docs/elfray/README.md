# ElfRay exports

The CSV files in this directory were generated with
[tt-elfray](https://github.com/tenstorrent/tt-elfray) from the compilation cache
produced on a P100 card by running:

```sh
pytest models/demos/vision/detection/rtdetr/tests/pcc.py::test_rtdetr_model
```

The test and ElfRay export were run against tt-metal commit
[`e1974e732afcb9339ab4329b863112e75d81ddd8`](https://github.com/tenstorrent/tt-metal/commit/e1974e732afcb9339ab4329b863112e75d81ddd8).

The resulting compute and data-movement reports are stored as `compute.csv.gz`
and `data movement.csv.gz`, respectively. They are gzip-compressed to keep the
files below the repository size limit. Extract them while retaining the
compressed files with:

```sh
gzip -dk -- *.csv.gz
```
