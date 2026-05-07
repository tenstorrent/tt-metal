import numpy as np
import soundfile as sf

audio, sr = sf.read("output.wav")
print(f"Sample rate: {sr}")
print(f"Audio shape: {audio.shape}")
print(f"Max amplitude: {np.max(audio)}")
print(f"Min amplitude: {np.min(audio)}")
print(f"Mean amplitude: {np.mean(audio)}")
print(f"RMS: {np.sqrt(np.mean(audio**2))}")
