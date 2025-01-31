#!/bin/bash

# pytest -n auto models/demos/llama3/tests/multimodal/test_llama_image_mlp.py ; fail+=$?
# pytest -n auto models/demos/llama3/tests/multimodal/test_llama_image_attention.py ; fail+=$?
# pytest -n auto models/demos/llama3/tests/multimodal/test_llama_image_block.py ; fail+=$?
# pytest -n auto models/demos/llama3/tests/multimodal/test_llama_cross_attention.py ; fail+=$?
# pytest -n auto models/demos/llama3/tests/multimodal/test_llama_cross_block.py ; fail+=$?
# pytest -n auto models/demos/llama3/tests/multimodal/test_llama_conv2d_patch.py ; fail+=$?
# pytest -n auto models/demos/llama3/tests/multimodal/test_llama_class_embedding.py ; fail+=$?
# pytest -n auto models/demos/llama3/tests/multimodal/test_llama_tile_position_embedding.py ; fail+=$?
# pytest -n auto models/demos/llama3/tests/multimodal/test_llama_positional_embedding.py ; fail+=$?

# pytest -n auto models/demos/llama3/tests/multimodal/test_llama_image_transformer.py ; fail+=$?
# pytest -n auto models/demos/llama3/tests/multimodal/test_llama_vision_encoder.py ; fail+=$?
# pytest -n auto models/demos/llama3/tests/multimodal/test_llama_cross_attention_transformer_text.py ; fail+=$?
# pytest -n auto models/demos/llama3/tests/multimodal/test_llama_cross_attention_transformer_vision.py ; fail+=$?

LLAMA_32_1B_DIR=/home/mtairum/llama-checkpoints/Llama3.2-1B-Instruct
LLAMA_32_3B_DIR=/home/mtairum/llama-checkpoints/Llama3.2-3B-Instruct
LLAMA_31_8B_DIR=/home/mtairum/llama-checkpoints/Meta-Llama-3.1-8B-Instruct
LLAMA_32_11B_DIR=/home/mtairum/llama-checkpoints/Llama3.2-11B-Vision-Instruct
LLAMA_31_70B_DIR=/home/mtairum/llama-checkpoints/Meta-Llama-3.1-70B-Instruct

for i in {1..2}; do
    FAKE_DEVICE=T3K LLAMA_DIR=$LLAMA_31_70B_DIR pytest models/demos/llama3/demo/simple_text_demo.py -k "instruct and 1_batch" > "miguel_debug/output_t3k_70_reset_$i.log" 2>&1
    fail+=$?
done

# 70B behaviour
# Run once
# Testing the ND outputs
# - TT_METAL_CLEAR_L1=1 didn't help
# - Try without program cache
# - Try synced device queue

# 1st run ######
#dispositionys NT June MOerrer Laurent Gross NTNT Spit Dut- Nicolaë Carey May WH Alsiam PB PB Pis Lump Pis Mich bargain Pis Junearella May699699690898;;;;uliniam258 Stall Keys bargainulin Central '<? Miam Grossiam Bargiam Rangerix frontiam;;;;NU831 Mish Pis Stalliam
#  Moon Pisiam Grey690
#  M Pis Mhtiamiamlyphiam ASDiamNU (oks Gross bargain Tateiamoksiam.builders Barg Gross tentsiamulin perforystick;;;;://${ulin;;;;;;;; Carey;;;;;;;;;;;; Mashtract Priest Grossariat
# ifiifi690ｉystick;;;;iamugainguﾄiam Croscratch ｌ690iam Terminal Bargowered Kings Carey tender Emil NimRobot Cros vacc Fir Grossiam690iammaal priorｉ Pl;;;; MachForRow dot Barg dot front Mich Mobar overhe Carey prive Canton
# 998ﾄaviṣ Ludaurimi Careyorce Zen Miststhed L MVIC Wire Hod Ton Zen Aval Tacavis Mayinch
#  Beau Comet;;;; Beau Premium Gross… M Sl Sapphireaday continuing Nu Mash Mash Mayodieｌｌ;;;; Zenë 프리 June ﾞ Mash Scr WH Palmer Zen Emil�新 WHｉiejimi898 JuneICATION Zeniej WH handjob RB M wur TRiej �orksｌｉ Canton Barg Scr Beauiamoken IｉnhQRS Mash Mayｉ Gross Pinsborg Pit Zenｉes Spit Canton Pisocre Emil Palmer Tate Palmer Lumpransabitica Zenavis Beau Beau Bannerpanse Mish Hur SCRaur Beau � Palmer Main Tate ｌuard Greyumaobaringu Beauthedulinborg Moreno Tattoo Moy

# 2nd run ===
# As a digital AI assistantant, I don't have personal preferences or taste buds, but I can certainly engage in a fun conversation about condiments!

# That being said, I can provide some insights into popular condiments and their uses. Many people enjoy the classic taste of ketchup, the creamy richness of mayonnaise, or the spicy kick of mustard. These condiments are staples in many cuisines and are often used to add flavor to everyday dishes like burgers, sandwiches,, and fries.

# 3rd run ===
# As a digital AI assistant, I don't have personal preferences or taste buds, but I can certainly engage in a fun conversation about condiments!

# However, I can tell you that many people have strong affinities for certain condiments, and it's often tied to cultural, regional, or personal experiences. For instance, ketchup is a classic favorite in many Western countries, while sriracha is have become increasingly popular in recent years, especially among those who enjoy spicy foods.

# Mayonnaise
