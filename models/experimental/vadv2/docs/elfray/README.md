# ElfRay exports

The CSV files in this directory were generated with
[tt-elfray](https://github.com/tenstorrent/tt-elfray) from the compilation cache
produced on a P100 card by running:

```sh
pytest models/experimental/vadv2/tests/pcc/test_tt_vad.py
```

The test and ElfRay export were run against tt-metal commit
[`fb8a11a93da6302edc6b89215f8a6cd0130ad84e`](https://github.com/tenstorrent/tt-metal/commit/fb8a11a93da6302edc6b89215f8a6cd0130ad84e).

The resulting compute and data-movement reports are stored as `compute.csv.gz`
and `data movement.csv.gz`, respectively. They are gzip-compressed to keep the
files below the repository size limit. Extract them while retaining the
compressed files with:

```sh
gzip -dk -- *.csv.gz
```
