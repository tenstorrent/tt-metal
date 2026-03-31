# Molmo2 Comprehensive Test Results

**Date:** 2026-03-31
**Platform:** T3K (8 devices)
**Branch:** ssinghal/Molmo2-8B_2

## Summary Table

| Test | Batch Size | Modality | Status | Coherence | Notes |
|------|------------|----------|--------|-----------|-------|
| Demo | 1 | Text | ✓ PASS | ✓ Coherent | Output: "4" (correct) |
| Demo | 1 | Image | ✓ PASS | ⚠ Minor | Minor repetition in output |
| Demo | 1 | Video (8 frames) | ✓ PASS | ✓ Coherent | 35 tok/s decode |
| Demo | 1 | Video (all frames) | ✗ FAIL | ✗ Garbage | Chunked processing issue |
| Demo | 32 | Text | ⚠ PARTIAL | ⚠ Mixed | ~40% correct, rest garbage/wrong |
| Demo | 4 | Image | ⚠ PARTIAL | ⚠ Mixed | 1/4 coherent, 3/4 repetitive |
| Server | 1 | Text | ✓ PASS | ✓ Coherent | Output: "4" |
| Server | 1 | Image | ✓ PASS | ✓ Coherent | Proper dog description |
| Server | 1 | Video | ✗ ERROR | - | Seq len 6974 > max_model_len 4096 |
| Server | 32 | Text (concurrent) | ✓ PASS | ✓ Coherent | 31/32 correct (97%) |
| Server | 4 | Image (concurrent) | ✗ FAIL | ✗ Garbage | Causes server state corruption |
| Server | 32 | Video | - | - | Not tested (seq len limit) |

## Detailed Results

### Demo Tests

#### 1. Demo Batch 1 - Text ✓ PASS
```
Prompt: "What is 2 + 2? Answer with just the number."
Output: "4"
TTFT: 373ms
Decode: 32.68 tok/s
Status: WORKING, COHERENT
```

#### 2. Demo Batch 1 - Image ✓ PASS
```
Prompt: "<|image|> What breed of dog is this? Answer briefly."
Output: "Based on the description, this appears to be a puppy appears to be a mixed breed."
TTFT: 2378ms (includes vision processing)
Decode: 35.47 tok/s
Status: WORKING, minor repetition ("appears to be...appears to be")
```

#### 3. Demo Batch 1 - Video (8 frames) ✓ PASS
```
Prompt: "<|video|> What letter did the person write first on the paper?"
Output: "The person wrote the letter 'M' on the letter 'A' on the paper."
Frames: 8
TTFT: 582ms
Decode: 35.08 tok/s
Status: WORKING, COHERENT (may not be factually perfect)
```

#### 4. Demo Batch 1 - Video (all 66 frames) ✗ FAIL
```
Prompt: "<|video|> What letter did the person write first on the paper?"
Output: "coffee, coffee, coffee, coffee..." (GARBAGE)
Frames: 66
TTFT: 5803ms
Decode: 33.14 tok/s
Status: FAILING - chunked vision processing produces garbage output
Issue: Cross-frame attention lost when processing >8 frames in chunks
```

#### 5. Demo Batch 32 - Text ⚠ PARTIAL
```
Test: Math questions "User N: What is N plus N+1?"
Results by user (expected vs actual):
- User 0: expected 3, got "32. What is 32 plus" - WRONG (copying other prompts!)
- User 1: expected 5, got 5 - CORRECT
- User 2: expected 7, got 7 - CORRECT
- User 3: expected 9, got 4 - WRONG
- User 4: expected 11, got "1111111111" - GARBAGE
- User 5: expected 13, got 13 - CORRECT
- User 6: expected 15, got 13 - WRONG
- User 7: expected 17, got 17 - CORRECT
- User 8: expected 19, got 1 - WRONG
- User 9-13: expected 21-29, got "2000000000" - GARBAGE
- User 18: expected 39, got 18 - WRONG

Approximate accuracy: ~40%
Status: PARTIAL - significant batch decode issues
Issue: Many users getting garbage or wrong answers - RoPE/attention batching issues
```

#### 6. Demo Batch 4 - Image ⚠ PARTIAL
```
Results:
- User 0: "This image shows a small dog on a skateboard. The dog on a skateboard. The dog is standing on a skateboard..." - REPETITIVE
- User 1: "The dog in the image appears to be a mixed breed...Labrador Retrieverseems to be a Labrador Retrieveralbich" - GARBLED
- User 2: "The dog in this image is standing on a skateboard." - COHERENT
- User 3: "Paws on wheels, a skateboard, he rides the breeze..." - SEMI-COHERENT (attempted haiku)

Throughput: 113.38 tok/s total (28.35 tok/s/user)
Status: PARTIAL - 1/4 coherent, 3/4 have issues
```

### Server Tests (vLLM API)

#### 7. Server Batch 1 - Text ✓ PASS
```
Request: {"messages": [{"role": "user", "content": "What is 2 + 2? Answer with just the number."}]}
Response: "4"
Status: WORKING, COHERENT
```

#### 8. Server Batch 1 - Image ✓ PASS
```
Request: Image + "What breed of dog is this? Answer briefly."
Response: "This appears to be a mixed breed dog, possibly with some Beagle or Jack Russell Terrier ancestry. The tricolor coat (brown, black, and white) and floppy ears are common in mixed breeds..."
Status: WORKING, COHERENT
```

#### 9. Server Batch 1 - Video ✗ ERROR
```
Request: Video + "What letter did the person write first on the paper?"
Error: "The decoder prompt (length 6974) is longer than the maximum model length of 4096"
Status: ERROR - Server max_model_len needs to be increased for video support
```

#### 10. Server Batch 32 - Text (Concurrent) ✓ PASS
```
Test: 32 concurrent math requests
Results:
  - Successful: 32/32
  - Correct answers: 31/32 (97%)
  - Total time: 19755ms
  - Avg latency: 19625ms

Sample responses:
  - User 0: expected 1, got "0 plus 1 is 1." ✓
  - User 10: expected 21, got "10 plus 11 is 21" ✓
  - User 20: expected 41, got "20 plus 21 is 41" ✓
  - User 31: expected 63, got "31 plus 32 is 63" ✓

Status: WORKING, HIGHLY COHERENT
```

#### 11. Server Batch 4 - Image (Concurrent) ✗ CRITICAL FAIL
```
Test: 4 concurrent image requests
Results:
  - All 4 responses: GARBAGE ("AokesokesableViewokes...")
  - After test: Server state corrupted, even text produces garbage
  - Required: Server restart + device reset

Status: CRITICAL FAILURE - Concurrent image requests corrupt server state
Issue: Likely memory corruption or state management bug in vLLM image handling
```

## Summary by Modality

### Text
| Test | Batch 1 | Batch 32 |
|------|---------|----------|
| Demo | ✓ PASS (coherent) | ⚠ PARTIAL (~40% correct) |
| Server Sequential | ✓ PASS (coherent) | N/A |
| Server Concurrent | ✓ PASS (coherent) | ✓ PASS (97% correct) |

**Conclusion:** Text works well in batch 1 and server concurrent. Demo batch 32 has RoPE/attention issues.

### Image
| Test | Batch 1 | Batch 4 |
|------|---------|----------|
| Demo | ✓ PASS (minor repetition) | ⚠ PARTIAL (1/4 coherent) |
| Server Sequential | ✓ PASS (coherent) | N/A |
| Server Concurrent | - | ✗ CRITICAL (corrupts server) |

**Conclusion:** Image works in batch 1. Batched/concurrent images have severe issues.

### Video
| Test | Batch 1 (8 frames) | Batch 1 (all frames) |
|------|---------------------|----------------------|
| Demo | ✓ PASS (coherent) | ✗ FAIL (garbage) |
| Server | ✗ ERROR (seq len) | ✗ ERROR (seq len) |

**Conclusion:** Video only works with limited frames (8). Chunked processing produces garbage. Server needs max_model_len increase.

## Known Issues

### Critical
1. **Server Concurrent Images:** Corrupts server state - all subsequent requests fail
2. **Video Chunked Processing:** >8 frames produces garbage output

### High
3. **Demo Batch 32 Text:** ~60% of outputs incorrect/garbage - RoPE/attention batching bug
4. **Demo Batch 4 Image:** 3/4 outputs have repetition/garbling

### Medium
5. **Server Video:** max_model_len=4096 too small for video (needs 8k-16k)
6. **Demo Batch 1 Image:** Minor repetition in output

## Performance Metrics

| Test | TTFT | Decode Speed |
|------|------|--------------|
| Demo Batch 1 Text | 373ms | 32.68 tok/s |
| Demo Batch 1 Image | 2378ms | 35.47 tok/s |
| Demo Batch 1 Video (8f) | 582ms | 35.08 tok/s |
| Demo Batch 4 Image | - | 113.38 tok/s (28.35/user) |
| Server Batch 32 Text | - | ~1.6 tok/s/request |

## Recommendations

### For Production Use
1. **Text:** Use server with concurrent requests (97% accuracy)
2. **Image:** Use batch 1 only (both demo and server sequential)
3. **Video:** Limit to 8 frames with `--max-video-frames 8`

### Required Fixes
1. Fix concurrent image handling in vLLM server (state corruption)
2. Fix demo batch 32 text coherence (RoPE/attention)
3. Fix or redesign chunked video processing
4. Increase server max_model_len for video support

### Docker Tests
- Not performed (would require server restart with docker)
- Based on server results, expect similar issues with concurrent images

## Test Environment
- TT-Metal: ssinghal/Molmo2-8B_2 branch
- Device: T3K (8x Wormhole B0)
- vLLM server: max_batch_size=32, max_seq_len=4096
