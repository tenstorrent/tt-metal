import os
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv


def _get_input_path(tmp_path: Path) -> Path:
    input_path = os.getenv("RVC_TEST_INPUT")
    if input_path:
        return Path(input_path)

    # Generate a short mono sine wave for a minimal smoke test.
    sr = 16000
    duration_sec = 1.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 440.0 * t)
    wav_path = tmp_path / "input.wav"
    sf.write(wav_path, audio, sr)
    return wav_path


def test_pipeline_inference_shapes(tmp_path: Path) -> None:
    load_dotenv()

    from rvc.vc.pipeline import Pipeline

    input_path = _get_input_path(tmp_path)

    pipe = Pipeline()

    audio_opt = pipe.infer(str(input_path), 0)

    assert audio_opt is not None
    assert audio_opt.ndim in (1, 2)
    assert audio_opt.shape[0] > 0
    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "output.wav"
    sf.write(output_path, np.asarray(audio_opt), pipe.tgt_sr)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    test_pipeline_inference_shapes(Path(tempfile.mkdtemp()))
