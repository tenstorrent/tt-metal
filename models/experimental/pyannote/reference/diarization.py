# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host steps of pyannote/speaker-diarization-3.1 after segmentation and embedding: speaker count,
clustering, reconstruction and speaker turns. Mirrors pyannote.audio 3.3 (SpeakerDiarization,
AgglomerativeClustering, Inference.aggregate, Binarize) in numpy/scipy, without pyannote.audio.

Windows may start at any time, so windows from separately processed audio chunks can be combined
on one global frame grid."""

import math

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist

from models.experimental.pyannote.reference.segmentation import FRAME_DURATION, FRAME_STEP, WINDOW_SECONDS

# speaker-diarization-3.1 clustering parameters (pipeline config.yaml).
THRESHOLD = 0.7045654963945799
MIN_CLUSTER_SIZE = 12
# WeSpeaker needs 400 samples of clean (non-overlapped) speech to use the clean mask.
MIN_CLEAN_SAMPLES = 400


def embedding_masks(segmentations: np.ndarray, window_samples: int = 160000) -> np.ndarray:
    """Binary segmentations [windows, frames, speakers] -> pooling masks [windows, speakers, frames].

    A speaker's frames that overlap no other speaker, when there are enough of them, else all its frames."""
    num_frames = segmentations.shape[1]
    min_frames = math.ceil(num_frames * MIN_CLEAN_SAMPLES / window_samples)
    clean = segmentations * (segmentations.sum(axis=2, keepdims=True) < 2)
    use_clean = clean.sum(axis=1, keepdims=True) > min_frames
    return np.where(use_clean, clean, segmentations).transpose(0, 2, 1).astype(np.float32)


def _start_frame(start: float) -> int:
    return int(np.rint(start / FRAME_STEP))


def _aggregate(scores: np.ndarray, starts: np.ndarray, skip_average: bool) -> np.ndarray:
    """Overlap-add of per-window scores [windows, frames, classes] onto one frame grid; NaN = absent."""
    num_windows, num_frames, num_classes = scores.shape
    total = _start_frame(starts.max() + WINDOW_SECONDS) + 1
    total = max(total, max(_start_frame(s) for s in starts) + num_frames)
    out = np.zeros((total, num_classes), np.float32)
    weight = np.zeros((total, num_classes), np.float32)
    seen = np.zeros((total, num_classes), np.float32)
    for start, score in zip(starts, scores):
        mask = 1.0 - np.isnan(score)
        f = _start_frame(start)
        out[f : f + num_frames] += np.nan_to_num(score, nan=0.0) * mask
        weight[f : f + num_frames] += mask
        seen[f : f + num_frames] = np.maximum(seen[f : f + num_frames], mask)
    average = out if skip_average else out / np.maximum(weight, 1e-12)
    average[seen == 0.0] = 0.0
    return average


def speaker_count(segmentations: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Number of active speakers per global frame [total frames]."""
    return np.rint(_aggregate(segmentations.sum(axis=-1, keepdims=True), starts, False)[:, 0]).astype(np.int64)


def _cluster(embeddings: np.ndarray) -> np.ndarray:
    """Centroid-linkage agglomerative clustering of unit-normalised embeddings, small clusters merged."""
    num = embeddings.shape[0]
    if num == 1:
        return np.zeros(1, np.int64)
    min_cluster_size = min(MIN_CLUSTER_SIZE, max(1, round(0.1 * num)))
    with np.errstate(divide="ignore", invalid="ignore"):
        normed = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
    clusters = fcluster(linkage(normed, method="centroid", metric="euclidean"), THRESHOLD, criterion="distance") - 1
    unique, counts = np.unique(clusters, return_counts=True)
    large, small = unique[counts >= min_cluster_size], unique[counts < min_cluster_size]
    if len(large) == 0:
        return np.zeros_like(clusters)
    if len(small) == 0:
        return clusters
    centroid = lambda ks: np.vstack([normed[clusters == k].mean(axis=0) for k in ks])
    for s, l in enumerate(np.argmin(cdist(centroid(large), centroid(small), metric="cosine"), axis=0)):
        clusters[clusters == small[s]] = large[l]
    return np.unique(clusters, return_inverse=True)[1]


def cluster(embeddings: np.ndarray, segmentations: np.ndarray) -> np.ndarray:
    """embeddings [windows, speakers, dim] -> global speaker per (window, speaker), -2 if inactive."""
    active = segmentations.sum(axis=1) > 0
    valid = ~np.any(np.isnan(embeddings), axis=2)
    w_idx, s_idx = np.where(active & valid)
    train = embeddings[w_idx, s_idx]
    labels = _cluster(train.copy())
    centroids = np.vstack([train[labels == k].mean(axis=0) for k in range(labels.max() + 1)])
    num_windows, num_speakers, dim = embeddings.shape
    distance = cdist(embeddings.reshape(-1, dim), centroids, metric="cosine").reshape(num_windows, num_speakers, -1)
    hard = np.argmax(2 - distance, axis=2)
    hard[~active] = -2
    return hard


def reconstruct(segmentations: np.ndarray, starts: np.ndarray, hard: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Per-frame global speaker activity [total frames, speakers]: the `count` most active speakers."""
    num_windows, num_frames, _ = segmentations.shape
    num_clusters = max(int(hard.max()) + 1, 1)
    clustered = np.full((num_windows, num_frames, num_clusters), np.nan)
    for w in range(num_windows):
        for k in np.unique(hard[w]):
            if k >= 0:
                clustered[w, :, k] = segmentations[w][:, hard[w] == k].max(axis=1)
    activations = _aggregate(clustered, starts, True)
    if activations.shape[1] < count.max():
        activations = np.pad(activations, ((0, 0), (0, count.max() - activations.shape[1])))
    n = min(len(activations), len(count))
    order = np.argsort(-activations[:n], axis=-1, kind="stable")
    binary = np.zeros_like(activations[:n])
    for t in range(n):
        binary[t, order[t, : count[t]]] = 1.0
    return binary


def turns(binary: np.ndarray) -> list[tuple[float, float, int]]:
    """Frame activity [frames, speakers] -> (start s, end s, speaker) turns, as pyannote's Binarize."""
    middle = np.arange(len(binary)) * FRAME_STEP + FRAME_DURATION / 2
    out = []
    for k in range(binary.shape[1]):
        active, start = binary[0, k] > 0.5, middle[0]
        for t, y in zip(middle[1:], binary[1:, k]):
            if active and y < 0.5:
                out.append((start, t, k))
                active, start = False, t
            elif not active and y > 0.5:
                active, start = True, t
        if active:
            out.append((start, middle[-1], k))
    return sorted(out)


def diarize(segmentations: np.ndarray, starts: np.ndarray, embeddings: np.ndarray) -> list[tuple[float, float, int]]:
    """Speaker turns from binary segmentations [windows, frames, 3], window starts [windows] in seconds
    and embeddings [windows, 3, dim] (from `embedding_masks`)."""
    starts = np.asarray(starts, dtype=np.float64)
    count = speaker_count(segmentations, starts)
    if count.max() == 0:
        return []
    hard = cluster(embeddings, segmentations)
    return turns(reconstruct(segmentations, starts, hard, count))
