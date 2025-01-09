#!/bin/bash

# Exit on any error
set -e

# Array to store latencies
declare -a latencies

# Array declaration using proper bash syntax
prompts=(
    "The sun rises slowly behind a perfectly plated breakfast scene. Thick, golden maple syrup pours in slow motion over a stack of fluffy pancakes, each one releasing a soft, warm steam cloud. A close-up of crispy bacon sizzles, sending tiny embers of golden grease into the air. Coffee pours in smooth, swirling motion into a crystal-clear cup, filling it with deep brown layers of crema. Scene ends with a camera swoop into a fresh-cut orange, revealing its bright, juicy segments in stunning macro detail."
    "The camera floats gently through rows of pastel-painted wooden beehives, buzzing honeybees gliding in and out of frame. The motion settles on the refined farmer standing at the center, his pristine white beekeeping suit gleaming in the golden afternoon light. He lifts a jar of honey, tilting it slightly to catch the light. Behind him, tall sunflowers sway rhythmically in the breeze, their petals glowing in the warm sunlight. The camera tilts upward to reveal a retro farmhouse with mint-green shutters, its walls dappled with shadows from swaying trees. Shot with a 35mm lens on Kodak Portra 400 film, the golden light creates rich textures on the farmer's gloves, marmalade jar, and weathered wood of the beehives."
    "Low-angle tracking shot, 18mm lens. The car drifts, leaving trails of light and tire smoke, creating a visually striking and abstract composition. The camera tracks low, capturing the sleek, olive green muscle car as it approaches a corner. As the car executes a dramatic drift, the shot becomes more stylized. The spinning wheels and billowing tire smoke, illuminated by the surrounding city lights and lens flare, create streaks of light and color against the dark asphalt. The cityscape – yellow cabs, neon signs, and pedestrians – becomes a blurred, abstract backdrop. Volumetric lighting adds depth and atmosphere, transforming the scene into a visually striking composition of motion, light, and urban energy."
    "cinematic shot of a female doctor in a dark yellow hazmat suit, illuminated by the harsh fluorescent light of a laboratory. The camera slowly zooms in on her face, panning gently to emphasize the worry and anxiety etched across her brow. She is hunched over a lab table, peering intently into a microscope, her gloved hands carefully adjusting the focus. The muted color palette of the scene, dominated by the sickly yellow of the suit and the sterile steel of the lab, underscores the gravity of the situation and the weight of the unknown she is facing. The shallow depth of field focuses on the fear in her eyes, reflecting the immense pressure and responsibility she bears."
)

# Function to calculate mean
calculate_mean() {
    local sum=0
    local count=${#latencies[@]}

    for latency in "${latencies[@]}"; do
        sum=$(echo "$sum + $latency" | bc)
    done

    echo "scale=2; $sum / $count" | bc
}

echo "Starting video generation..."

# Proper bash for loop syntax with timing
for i in "${!prompts[@]}"; do
    start_time=$(date +%s.%N)

    MOCHI_DIR=/proj_sw/mochi-data FAKE_DEVICE=T3K python models/experimental/mochi/cli_tt.py \
        --prompt "${prompts[$i]}" \
        --model_dir "models/mochi/mochi-v1-128x96"  || exit 1

    end_time=$(date +%s.%N)
    latency=$(echo "$end_time - $start_time" | bc)
    latencies+=($latency)

    echo "Iteration $((i+1)) completed in ${latency} seconds"
done

# Print summary
echo -e "\nLatency Summary:"
for i in "${!latencies[@]}"; do
    echo "Video $((i+1)): ${latencies[$i]} seconds"
done

mean_latency=$(calculate_mean)
echo -e "\nMean latency: ${mean_latency} seconds"
