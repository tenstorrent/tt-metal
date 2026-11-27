# Reference-media decoder access controls

CPU integration tests use real PyAV, synthetic embedded media and temporary
canaries. They import the actual pure-CPU MiniMax-H3 modules without importing
the device-dependent `pipelines` package initializer. Run with PyAV, NumPy,
CPU PyTorch and Pillow installed:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -B tests/reference_media_cpu/test_decoders.py -v
```

The LTX requirements pin PyAV 17.0.1; that wheel uses FFmpeg 8.0.1. The tests
also pass with PyAV 17.1.0 / FFmpeg 8.1.1. This is CPU decoder coverage only,
not native, model-quality, serving or deployment qualification.

The reported MOV alias vulnerability was **not reproduced with defaults**.
FFmpeg's MOV demuxer defaults `enable_drefs` and `use_absolute_path` to false.
The fixtures replace both tracks' `dref` entries with real QuickTime `alis`
records and erase embedded sample data. Relative and absolute aliases decode
the canary only with the necessary opt-ins; defaults do not open the canary.
Positive controls prevent a malformed fixture from masquerading as protection.

A separate, demonstrated gap is automatic probing of concat and HLS manifests.
Both original decoder functions read and decode another local file referenced
by these manifests. Concat's default `safe=1` accepts relative filenames, and
`protocol_whitelist=file` still permits the read. Filename extensions do not
prevent probing a concat manifest disguised as `.mov`.

The decoder now opens the supplied local file with Python and gives PyAV that
file object. An empty FFmpeg protocol whitelist rejects protocol-backed child
opens, including `file`; a rejecting `io_open` callback denies demuxer requests
for secondary resources. Concat creates child contexts without inheriting the
callback, so the whitelist is necessary. No MOV options enabling references
are exposed. Ordinary embedded MOV, MP4/AAC, Matroska and WAV are covered,
including both the video's second-pass soundtrack and standalone audio decode.

URLs, concat/HLS playlists and external-track containers are unsupported.
The tested reference forms are local concat/HLS paths and relative/absolute
MOV aliases; top-level `file:` and `concat:` protocol strings are also rejected.
No network targets are contacted by the tests. The empty whitelist denies all
FFmpeg protocols, but these tests do not certify every demuxer or native parser,
protect against native-code vulnerabilities, or enforce file-root, symlink,
file-size, duration or resource-budget policies on the caller's selected file.
MOV `url`/`urn` drefs are not covered by these fixtures (the inspected demuxer
interprets `alis` references). Callers retain responsibility for authorizing
the top-level local path and enforcing upload budgets.

Primary implementation sources:

- [FFmpeg 8.0.1 MOV defaults and alias handling](https://github.com/FFmpeg/FFmpeg/blob/n8.0.1/libavformat/mov.c)
- [FFmpeg 8.0.1 concat child contexts and safe paths](https://github.com/FFmpeg/FFmpeg/blob/n8.0.1/libavformat/concatdec.c)
- [FFmpeg 8.0.1 protocol whitelist enforcement](https://github.com/FFmpeg/FFmpeg/blob/n8.0.1/libavformat/avio.c)
- [PyAV 17.0.1 file-object and custom I/O support](https://github.com/PyAV-Org/PyAV/blob/v17.0.1/av/container/core.py)
- [FFmpeg demuxer documentation](https://ffmpeg.org/ffmpeg-formats.html)
