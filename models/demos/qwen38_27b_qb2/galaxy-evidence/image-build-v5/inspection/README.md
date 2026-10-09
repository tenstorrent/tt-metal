# OCI archive inspection

The completed second inspection verifies the manifest/config digests and all
46 declared compressed layer digests/sizes, then counts the fully decompressed
layer tar bytes without extracting files. The image combines zstd base layers
with gzip-generated layers. The first inspection assumed gzip and failed on
the first zstd layer; the corrected inspector dispatches by OCI media type.
The image itself was not changed.

Compressed layers total 6,010,425,345 bytes; decompressed layer tar streams
total 18,909,817,344 bytes. Their sum plus an 8-GiB reserve exceeds available
disk on the build host, so no Docker import was attempted there. The queued
runtime check will copy the archive to the allocated Qwen host and recheck
space immediately before importing. These size/digest checks do not open
hardware or validate inference.
