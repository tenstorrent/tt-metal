#!/usr/bin/env python3
"""
Demo script to simulate training updates to output.txt and validation.txt
This helps test the Streamlit dashboard without running actual training.
"""

import time
import random
import math


def simulate_training(max_steps=1000, start_step=0):
    """Simulate training progress by writing to output.txt and validation.txt"""

    print("Starting training simulation...")
    print("This will write to output.txt and validation.txt")
    print("Open the Streamlit dashboard to see live updates!")
    print("Press Ctrl+C to stop\n")

    step = start_step
    epoch = 1

    # Initial learning rate
    lr_max = 3e-4
    lr_min = 1e-5
    warmup_steps = 20

    try:
        while step < max_steps:
            # Calculate learning rate (with warmup and decay)
            if step < warmup_steps:
                lr = lr_max * (step / warmup_steps)
            else:
                # Cosine decay
                progress = (step - warmup_steps) / (max_steps - warmup_steps)
                lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * progress))

            # Simulate decreasing loss with some noise
            progress_factor = 1 - (step / max_steps)
            train_loss = max(0.05, 2.5 * progress_factor**0.7 + random.uniform(-0.15, 0.1))
            val_loss = max(0.05, 2.7 * progress_factor**0.7 + random.uniform(-0.15, 0.1))

            # Write to output.txt
            with open("output.txt", "w") as f:
                f.write(
                    f"LR: {lr:.2e}, training_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, step: {step}, epoch: {epoch}\n"
                )

            # Update validation.txt every 50 steps
            if step % 50 == 0:
                accuracy = min(95, 60 + (step / max_steps) * 30 + random.uniform(-3, 3))

                questions = [
                    ("What is 15 + 27?", "42", "15 + 27 = 42"),
                    ("If John has 5 apples and buys 8 more, how many does he have?", "13", "5 + 8 = 13"),
                    ("Sarah has $20 and spends $7. How much money does she have left?", "$13", "$20 - $7 = $13"),
                    ("A train travels 60 miles in 2 hours. What is its average speed?", "30 mph", "60 / 2 = 30 mph"),
                    ("What is 100 - 47?", "53", "100 - 47 = 53"),
                    ("If a book costs $12 and you buy 3, what's the total?", "$36", "12 × 3 = $36"),
                ]

                val_content = f"Validation Results - Step {step}, Epoch {epoch}\n"
                val_content += "=" * 50 + "\n\n"

                for q, expected, answer in random.sample(questions, 4):
                    val_content += f"Question: {q}\n"
                    val_content += f"Expected: {expected}\n"
                    val_content += f"Model Answer: {answer}\n\n"

                val_content += "=" * 50 + "\n"
                val_content += f"Validation Metrics:\n"
                val_content += f"- Validation Loss: {val_loss:.4f}\n"
                val_content += f"- Accuracy: {accuracy:.1f}%\n"
                val_content += f"- Total Examples: 4\n"
                val_content += f"- Correct: {int(4 * accuracy / 100)}\n"
                val_content += "=" * 50 + "\n"

                with open("validation.txt", "w") as f:
                    f.write(val_content)

            # Progress output
            if step % 10 == 0:
                print(f"Step {step:4d}/{max_steps} | Loss: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.2e}")

            # Increment step
            step += 1

            # Update epoch
            if step % 100 == 0:
                epoch += 1

            # Sleep to simulate training time
            time.sleep(0.5)  # Update every 0.5 seconds for demo purposes

    except KeyboardInterrupt:
        print("\n\nTraining simulation stopped by user")
        print(f"Final step: {step}")

    print("\nSimulation complete!")


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    max_steps = 1000
    start_step = 0

    if len(sys.argv) > 1:
        max_steps = int(sys.argv[1])
    if len(sys.argv) > 2:
        start_step = int(sys.argv[2])

    print(f"Max steps: {max_steps}")
    print(f"Starting from step: {start_step}\n")

    simulate_training(max_steps, start_step)
