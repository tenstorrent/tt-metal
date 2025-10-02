#!/usr/bin/env python3
"""
Test the Fixed Mandelbrot Math
"""


def test_mandelbrot_point(cx, cy, max_iter=100):
    """Test a single Mandelbrot point"""
    zx, zy = 0.0, 0.0
    for i in range(max_iter):
        zx2, zy2 = zx * zx, zy * zy
        if zx2 + zy2 > 4.0:
            return i
        zx, zy = zx2 - zy2 + cx, 2 * zx * zy + cy
    return max_iter


def test_kernel_coordinate_mapping():
    """Test the coordinate mapping that the kernels are now using"""
    print("🧮 Testing Fixed Kernel Coordinate Mapping")
    print("=" * 50)

    # Kernel parameters
    IMAGE_WIDTH = 512
    IMAGE_HEIGHT = 512
    x_min, x_max = -2.5, 1.5
    y_min, y_max = -2.0, 2.0

    dx = (x_max - x_min) / IMAGE_WIDTH
    dy = (y_max - y_min) / IMAGE_HEIGHT

    print(f"Image: {IMAGE_WIDTH}×{IMAGE_HEIGHT}")
    print(f"Complex plane: x[{x_min}, {x_max}] y[{y_min}, {y_max}]")
    print(f"Deltas: dx={dx:.7f} dy={dy:.7f}")
    print()

    # Test some key pixels that should show interesting behavior
    test_pixels = [
        # Center of image (should be near origin)
        (256, 256, "Center of image"),
        (256, 200, "Above center"),
        (256, 312, "Below center"),
        # Points that should be in the Mandelbrot set
        (320, 256, "Right of center"),
        (384, 256, "Further right"),
        (200, 256, "Left of center"),
        # Edge cases
        (0, 256, "Left edge"),
        (511, 256, "Right edge"),
        (256, 0, "Top edge"),
        (256, 511, "Bottom edge"),
    ]

    print("Testing Key Pixels:")
    print("Pixel(x,y) → Complex(cx,cy) → Iterations → Status")
    print("-" * 60)

    for x, y, desc in test_pixels:
        # Use FIXED coordinate mapping
        cx = x_min + x * dx
        cy = y_max - y * dy  # FIXED: y_max - y*dy

        iterations = test_mandelbrot_point(cx, cy, 100)

        if iterations >= 100:
            status = "IN SET"
        elif iterations <= 2:
            status = "ESCAPES FAST"
        else:
            status = f"ESCAPES in {iterations}"

        print(f"Pixel({x:3d},{y:3d}) → c({cx:7.3f},{cy:6.3f}) → {iterations:3d} → {status:12s} ({desc})")

    print()

    # Test some specific Mandelbrot set points by finding their pixel coordinates
    print("Testing Known Mandelbrot Points:")
    print("Complex(cx,cy) → Pixel(x,y) → Iterations → Status")
    print("-" * 60)

    known_points = [
        (0.0, 0.0, "Origin"),
        (-0.5, 0.0, "Real axis point"),
        (-1.0, 0.0, "Real axis point"),
        (-0.7269, 0.1889, "Known interior point"),
        (0.25, 0.0, "Real axis - boundary"),
        (-0.75, 0.1, "Near main bulb"),
    ]

    for cx, cy, desc in known_points:
        # Find corresponding pixel
        x = int((cx - x_min) / dx)
        y = int((y_max - cy) / dy)  # FIXED mapping

        # Verify it's in bounds
        if 0 <= x < IMAGE_WIDTH and 0 <= y < IMAGE_HEIGHT:
            iterations = test_mandelbrot_point(cx, cy, 100)

            if iterations >= 100:
                status = "IN SET"
            elif iterations <= 2:
                status = "ESCAPES FAST"
            else:
                status = f"ESCAPES in {iterations}"

            print(f"c({cx:7.3f},{cy:6.3f}) → Pixel({x:3d},{y:3d}) → {iterations:3d} → {status:12s} ({desc})")
        else:
            print(f"c({cx:7.3f},{cy:6.3f}) → OUT OF BOUNDS → --- → ------------ ({desc})")


def main():
    test_kernel_coordinate_mapping()

    print()
    print("🎯 Analysis:")
    print("• Y-coordinate mapping is now FIXED")
    print("• Points at x=0 (cx=-2.5) will always escape quickly (outside set)")
    print("• Points near center of image (x=256, y=256) should show interesting behavior")
    print("• The kernel debug shows edge pixels, but center pixels have the real action!")


if __name__ == "__main__":
    main()
