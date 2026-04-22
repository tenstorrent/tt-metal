from PIL import Image, ImageDraw

img = Image.new("RGB", (400, 200), "white")
draw = ImageDraw.Draw(img)

text = "DOTS OCR TEST\nInvoice 123\nTotal: 450"

# Draw dotted text manually
x, y = 10, 20
for char in text:
    if char == "\n":
        y += 30
        x = 10
        continue

    # draw dots instead of solid text
    for dx in range(0, 10, 2):
        for dy in range(0, 10, 2):
            draw.point((x + dx, y + dy), fill="black")

    x += 12

img.save("dots_sample.jpg")
