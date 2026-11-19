# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only diverse teacher-forced batch inputs; no generation or quality verdict.

Each authored paragraph supplies one independent row. Text is tokenized without
special tokens, padding, repetition, or a chat template. Upstream HF layers see
the prefill followed by every held-out token through the same causal cache.
"""

import argparse
import gc
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests.prepare_optimized_activations import embedding_inputs

PROMPTS = [
    """The orchard keeper walked between the pear trees before sunrise, counting branches that needed support. Last week's wind had loosened three wooden stakes near the western fence. She marked those trees with blue ribbon, then checked the irrigation channels for fallen leaves. Her notebook distinguished urgent repairs from work that could wait until the next dry afternoon.""",
    """A student measuring the cooling of tea placed identical cups on a wooden table. One cup had a lid, another stood beside an open window, and the third remained untouched near the wall. He recorded temperature every two minutes and noted the room temperature separately. Before comparing the curves, he checked that each cup had started with the same volume.""",
    """Inside the railway archive, a volunteer opened a box of handwritten timetables. Several pages described a station that no longer appeared on modern maps. She compared dates, distances, and platform numbers to reconstruct its location. A penciled note about a flooded bridge explained why one summer schedule listed a longer route through the neighboring valley.""",
    """For the community kitchen, Luis planned a vegetable stew using carrots, beans, tomatoes, and barley. The dried beans needed soaking overnight, while the barley could cook directly in the broth. He wrote a sequence of preparation steps so volunteers arriving at different times could contribute. The final tasting would determine whether the pot needed more herbs or salt.""",
    """The astronomy club gathered on a hillside with notebooks and a small telescope. Clouds covered the eastern horizon, but the western sky remained clear enough to identify several bright stars. One observer recorded the time while another drew their relative positions. They planned to return on the following evening and compare how the same pattern appeared above the trees.""",
    """A bicycle mechanic listened to a clicking sound from the rear wheel before removing any parts. The noise appeared once per rotation and became louder under load. She inspected the spokes, checked the brake clearance, and examined the tire for debris. Only after those simple checks did she consider opening the hub to inspect its bearings.""",
    """At the museum workshop, children built bridges from paper and string. Each team received the same amount of material and tested its design with small wooden blocks. A folded beam held more weight than a flat strip, which prompted questions about shape and stiffness. The instructor encouraged everyone to sketch the failed designs as carefully as the successful ones.""",
    """The coastal survey team returned to a beach after a week of strong tides. They measured the distance from fixed markers to the waterline and photographed changes in the dunes. New shells had accumulated near the northern path, while a shallow channel crossed the southern end. Their report separated short-term weather effects from changes that might persist across seasons.""",
    """In a quiet rehearsal room, the violinist practiced a difficult passage at half speed. She isolated the shift between two notes, repeated it with different bow lengths, and listened for uneven tone. When the movement felt comfortable, she restored the surrounding phrase. Her practice journal recorded the tempo that remained reliable rather than the fastest attempt of the morning.""",
    """The software team investigated why a nightly report sometimes omitted the final row. They compared the input files, inspected the loop boundaries, and reproduced the issue with a small example. The error occurred only when the file lacked a trailing newline. A test with three short records made the problem clear and provided a focused way to verify the correction.""",
    """A gardener compared two patches of soil before planting lettuce. One patch stayed damp after rain, while the other dried quickly in the afternoon sun. She mixed compost into both and placed identical seedlings at equal distances. Over the next month, she planned to record leaf growth, watering amounts, and any signs of damage from insects.""",
    """The translator studied a letter whose meaning depended on an unfamiliar regional expression. A literal translation sounded oddly formal, so she consulted neighboring sentences and an older dictionary. The writer seemed to be making a friendly invitation rather than issuing an instruction. She drafted two alternatives and asked a colleague which one preserved the tone of the original.""",
    """During a school debate, one group proposed extending library hours on weekends. They collected attendance figures, interviewed students, and estimated staffing needs. Another group suggested improving the online catalog first. The moderator asked both sides to explain which problem their proposal solved and what evidence would show that the change had helped readers after three months.""",
    """The potter opened the kiln after it had cooled overnight. Most bowls showed the expected green glaze, but two had uneven brown streaks near their rims. She compared their positions in the kiln with her firing notes. The results suggested that placement and airflow mattered as much as the glaze mixture, so she planned a smaller controlled firing.""",
    """A hiker studied the trail map before leaving the shelter. The direct route crossed a stream that could rise after heavy rain, while the longer path followed a ridge with little shade. She checked the forecast, refilled her water bottle, and wrote her planned destination in the logbook. Her choice balanced daylight, weather, and the condition of the group.""",
    """The accountant reconciled a small shop's receipts against its bank deposits. Most entries matched immediately, but a weekend payment appeared two days later than expected. She checked the transaction identifier before treating it as a missing sale. By preserving both the sale date and settlement date, the spreadsheet could explain the difference without changing the original records.""",
    """At a birdwatching station, volunteers counted migrating geese in short observation intervals. Large flocks were divided into visual groups to reduce counting errors. Each observer recorded wind direction and visibility alongside the estimated number of birds. When two counts differed greatly, they discussed the viewing conditions instead of simply averaging the numbers and hiding the disagreement.""",
    """A carpenter repaired a drawer that stuck whenever the weather became humid. He measured the gaps along both sides and checked whether the frame remained square. The wood had swollen slightly near the back corner. Rather than removing material everywhere, he marked the contact point with chalk and adjusted that small area before testing the drawer again.""",
    """The water laboratory received samples from three wells after a period of heavy rain. Technicians labeled each bottle, recorded collection times, and included a clean control sample. They repeated any reading that fell outside the instrument's calibration range. The final table kept measurements separate from interpretations so another analyst could review how the conclusions had been reached.""",
    """In the theater's costume room, a designer organized fabric scraps by texture and color. A heavy wool coat needed to look convincing under bright stage lights without making the actor uncomfortable. She tested a lighter lining and adjusted the seams to allow broader movement. Photographs from the rehearsal helped reveal details that were difficult to judge on a hanger.""",
    """A chess coach asked the players to explain their plans before moving a piece. One student wanted to attack the king immediately, while another preferred improving the position of a quiet knight. The coach set up both continuations and examined the opponent's strongest replies. The exercise rewarded careful comparison of alternatives instead of choosing the first attractive move.""",
    """The bakery tested a new bread recipe with two fermentation times. Both batches used flour from the same bag, equal amounts of yeast, and water at the same temperature. The baker measured loaf height and photographed the crumb after cooling. Customer comments were recorded separately because a preference for flavor did not necessarily match the loaf with the greatest volume.""",
    """A geologist examined layers exposed in a roadside cutting. Fine sand lay above darker clay, and several rounded stones formed a narrow band between them. She sketched the sequence before collecting samples. Nearby roots and recent erosion could distort the surface, so she compared several points along the cutting rather than relying on one especially clear-looking section.""",
    """The town council considered where to place new benches in the public square. Shopkeepers preferred locations near the entrances, while older residents asked for shade and a clear view of the bus stop. A temporary arrangement allowed people to try several positions. Observations over a week revealed which benches were used and where pedestrian traffic became crowded.""",
    """A mathematics teacher introduced fractions by dividing sheets of colored paper. Students first made halves and quarters, then compared pieces from sheets of different sizes. The activity revealed why the size of the whole must be specified. Later, the class used number lines to represent the same fractions without depending on the dimensions of a particular sheet.""",
    """The wildlife camera captured a fox near the forest path shortly before midnight. A second image showed only a moving tail at the edge of the frame. Researchers checked timestamps and battery status before estimating how many visits had occurred. They kept uncertain sightings in a separate column so that later analysis would not treat every blurred shape as a new animal.""",
    """An engineer reviewed the energy use of a small office building. Heating demand rose on cold mornings, but some equipment continued drawing power after everyone had left. She compared meter readings with occupancy records and maintenance schedules. The first proposed change was a simple timer, followed by measurements to determine whether the overnight consumption actually decreased.""",
    """The bookbinder repaired a volume whose pages had loosened near the middle. She examined the original stitching and tested the paper with a soft brush before choosing an adhesive. A repair that looked strong could still make the book difficult to open. Her aim was to support the damaged section while preserving the flexibility of the surrounding pages.""",
    """At the swimming pool, a coach timed short intervals for four athletes. Each swimmer rested for the same duration between attempts, and the lane assignments rotated to reduce any advantage from the pool layout. The coach noted technique changes alongside the stopwatch readings. A faster time was useful only if the movement remained controlled and could be repeated consistently.""",
    """A local historian compared oral accounts of an old ferry crossing. Different residents remembered different landing places, partly because the riverbank had shifted over time. She placed each account on a dated map and looked for supporting photographs. The resulting timeline showed that the apparently conflicting stories could describe different periods rather than mistakes in memory.""",
    """The robotics students programmed a small vehicle to follow a dark line across the floor. Bright sunlight near the window confused its sensor, causing sharp turns at one corner. They logged the sensor values and tried a calibration step before changing the steering rule. Testing under both shaded and sunny conditions helped separate an input problem from a control problem.""",
    """A seed library organized packets donated by gardeners at the end of summer. Volunteers recorded the plant variety, harvest year, and growing conditions whenever those details were available. Some packets contained too few seeds for distribution, so they were reserved for a small germination trial. The catalog clearly distinguished confirmed information from notes that still needed checking.""",
]


def run(args):
    import torch
    from transformers import AutoTokenizer
    from transformers.cache_utils import DynamicCache

    from models.autoports.qwen_qwen3_6_35b_a3b.tests import reference

    assert "ttnn" not in sys.modules, "This preparation must never import TTNN"
    torch.set_num_threads(args.threads)
    total = args.prefill_length + args.decode_steps
    assert args.prefill_length > 0 and args.decode_steps > 0
    assert len(PROMPTS) == len(set(PROMPTS)) == 32
    layers = sorted(set(args.layers))
    cfg = reference.config()
    assert 0 <= layers[0] <= layers[-1] < cfg.num_hidden_layers
    assert total <= cfg.max_position_embeddings
    output = args.output.resolve()
    metadata_path = output.with_suffix(".metadata.json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("Use fresh artifact paths to preserve prior evidence")
    tokenizer = AutoTokenizer.from_pretrained(reference.SNAPSHOT, local_files_only=True)
    all_ids = [tokenizer.encode(text, add_special_tokens=False) for text in PROMPTS]
    assert min(map(len, all_ids)) >= total, "Authored prompts must be long enough without padding or repetition"
    input_ids = torch.tensor([row[:total] for row in all_ids], dtype=torch.long)
    assert len({tuple(row.tolist()) for row in input_ids}) == 32
    embedding_rows, embedding_metadata = [], []
    for row in input_ids:
        hidden, metadata = embedding_inputs(reference.SNAPSHOT, row.unsqueeze(0), cfg.hidden_size)
        embedding_rows.append(hidden)
        embedding_metadata.append(metadata)
    hidden = torch.cat(embedding_rows, dim=0)
    prefix = hidden[:, : args.prefill_length].contiguous()
    heldout = [hidden[:, pos : pos + 1].contiguous() for pos in range(args.prefill_length, total)]
    del hidden, embedding_rows
    captured, upstream = {}, []
    started = time.monotonic()
    for index in range(layers[-1] + 1):
        assert prefix.dtype == torch.bfloat16 and torch.isfinite(prefix).all()
        assert all(t.dtype == torch.bfloat16 and torch.isfinite(t).all() for t in heldout)
        if index in layers:
            captured[str(index)] = {"x": prefix.clone(), "xd": torch.stack(heldout)}
        if index == layers[-1]:
            break
        state_dict = reference.weights(index)
        hf = reference.layer(state_dict, index)
        del state_dict
        cache = DynamicCache(config=cfg)
        prefix = reference.hf_forward(hf, prefix, cache, 0)
        heldout = [reference.hf_forward(hf, t, cache, args.prefill_length + step) for step, t in enumerate(heldout)]
        upstream.append({"layer": index, "kind": cfg.layer_types[index]})
        del cache, hf
        gc.collect()
        print(f"HF_BATCH_UPSTREAM_DONE layer={index} elapsed_s={time.monotonic() - started:.3f}", flush=True)
    assert "ttnn" not in sys.modules
    metadata = {
        "schema_version": 1,
        "purpose": "multichip decoder teacher-forced diverse routing inputs; no generation or quality verdict",
        "model": reference.MODEL,
        "checkpoint_revision": reference.REVISION,
        "snapshot": str(reference.SNAPSHOT),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": sys.argv,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "safetensors")},
        "prompt_format": "raw authored natural-text teacher forcing; no chat template or special tokens",
        "chat_template_applied": False,
        "generation_performed": False,
        "padding_or_repetition": False,
        "source_prompts": PROMPTS,
        "source_token_counts": [len(row) for row in all_ids],
        "selected_text_decoded": [tokenizer.decode(row.tolist(), skip_special_tokens=False) for row in input_ids],
        "input_ids": input_ids.tolist(),
        "batch": 32,
        "prefill_length": args.prefill_length,
        "decode_steps": args.decode_steps,
        "held_out_decode_positions": list(range(args.prefill_length, total)),
        "layers": layers,
        "layer_kinds": {str(idx): cfg.layer_types[idx] for idx in layers},
        "propagated_upstream_layers": upstream,
        "propagation": "At each upstream layer: batch prefill through fresh causal cache, then held-out tokens sequentially through that same cache",
        "attention_implementation": cfg._attn_implementation,
        "embedding_rows": embedding_metadata,
        "dtype": "torch.bfloat16",
        "device": "CPU",
        "cpu_threads": args.threads,
        "ttnn_imported": False,
        "elapsed_seconds": time.monotonic() - started,
        "source_files": [str(Path(__file__).resolve()), str(Path(reference.__file__).resolve())],
        "tensor_shapes": {idx: {name: list(t.shape) for name, t in values.items()} for idx, values in captured.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"input_ids": input_ids, "layers": captured, "metadata": metadata}, output)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"HF_BATCH_ACTIVATIONS_READY {output} layers={layers} batch=32", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-length", type=int, default=32)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 3])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
